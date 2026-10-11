//! VENDORED from egglog-experimental PR #60 (branch `unstable-subst`,
//! head eaf27ca, author oflatt), with documented vendoring deltas. Vendored
//! rather than depended-on while the PR is unmerged (ruling: take
//! exactly the audited contract; drop this file for the upstream crate
//! once it merges). Registered by `new_egraph()` in egglog_snippet.rs.
//!
//! Performance delta: constructor signatures, output-sort indices and declared
//! input storage kinds use egglog's shared immutable schema catalogue. Typed
//! primitive registrations also supply the known root sort to the walk. Table
//! registration and type declaration changes invalidate it; live rows, unions,
//! container contents and substitution results are never cached.
//!
//! Substitution over a reachable sub-e-graph: the `unstable-subst` primitive.
//!
//! `(unstable-subst root map)` takes an e-class `root` of any eq-sort and a
//! `Map` whose key and value sorts are the same eq-sort. It walks the
//! constructor rows reachable from `root`, copies the part of that sub-e-graph
//! the substitution actually touches while replacing each key e-class with its
//! mapped value, and returns the e-class of the copied root.
//!
//! The walk follows constructor rows only — `function` rows are analyses over
//! the term structure, not part of it — and reaches through container-valued
//! children, rebuilding them with substituted contents.
//!
//! Unaffected e-classes are shared with the original rather than copied, so
//! substituting an empty map returns `root` itself and writes nothing.
//!
//! # Warning: `root` must already exist
//!
//! Pass a root the rule's query bound, or one from an earlier command — not a
//! term the enclosing action just built.
//!
//! The walk reads committed table contents, and an action's writes stay staged
//! until it finishes. A root this action built has no rows yet, so the walk
//! finds no e-nodes under it, nothing is affected, and it comes back
//! **unchanged and without an error**. `(unstable-subst (Mul x y) m)`, building
//! its own argument, silently does nothing. The same applies to any term under
//! the root that the action just built: it is not there to be substituted.
//!
//! Replacements are not affected — a map's values are spliced into the copy
//! without being walked, so those can be built in the same action.
//!
//! # Other properties worth knowing
//!
//! - The region's equations are substituted along with its terms. Copying an
//!   e-class copies every one of its e-nodes, so `t1 = t2` in the original
//!   becomes `σ(t1) = σ(t2)` in the copy — and an e-node with no substituted
//!   children copies to itself, merging the copy back into the original class.
//!   That is correct for equations that hold for every value of the substituted
//!   classes (anything a rewrite rule derived) and wrong for a ground `union`
//!   pinning one of them down, so only substitute classes that behave like
//!   universally quantified variables.
//! - The snapshot comes from live table contents, so this is a `Context::Full`
//!   primitive: top-level actions and `:naive` rule heads only.
//!
//! Copies are named by `lookup_or_insert`, the same way `(Add a b)` in an
//! action is, so no e-class id is ever invented here. A cyclic e-class can
//! therefore only be copied if it has an e-node whose children all lie outside
//! the cycle to name it first — `x = {Var "x", Add x (Num 0)}` does, and works.
//! A cycle with no such e-node is an error rather than a silent partial copy.

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use egglog::api::RawValues;
use egglog::ast::Span;
use egglog::constraint::{self, Constraint, ImpossibleConstraint, TypeConstraint};
use egglog::sort::MapContainer;
use egglog::{
    ArcSort, Atom, AtomTerm, ConstructorCatalog, Core, EGraph, Error, FullPrim, FullState,
    Primitive, Read, SortStorageKind as Kind, TypeInfo, Value, Write,
};

/// The name of the primitive, as written in an egglog program.
pub const SUBST: &str = "unstable-subst";

// Storage classification comes from the immutable declared-sort catalogue.
// Container-valued contents are still inspected dynamically during each walk.
fn kind_of(sort: &ArcSort) -> Kind {
    Kind::of(sort)
}

/// One reachable constructor row, keyed by the constructor it came from so the
/// build pass can re-apply it.
struct ENode {
    ctor: usize,
    children: Vec<Value>,
    /// Copied rows of subsumed sources are subsumed in turn (vendoring
    /// delta): our identity folds retire a class's grounding spellings
    /// by :subsume, so skipping them entirely (upstream behavior)
    /// reports spurious ungrounded cycles. Subsumption is a matching
    /// fence, not semantic non-membership — the copy keeps the fence.
    subsumed: bool,
}

/// The reachable sub-e-graph.
#[derive(Default)]
struct Snapshot {
    /// E-nodes of each reachable e-class, including subsumed rows — their
    /// copies are subsumed in turn, so nothing is resurrected live.
    nodes: HashMap<Value, Vec<ENode>>,
    /// Contents of each reachable container value, with the [`TypeId`] to
    /// rebuild it under.
    containers: HashMap<Value, (TypeId, Vec<(ArcSort, Value)>)>,
    /// The e-classes each e-class references, flattened through containers.
    /// Drives both the affected fixpoint and the build order.
    deps: HashMap<Value, Vec<Value>>,
}

struct Walk<'a> {
    catalog: &'a ConstructorCatalog,
    skip: &'a HashSet<String>,
    map: &'a BTreeMap<Value, Value>,
    snapshot: Snapshot,
    /// E-classes the substitution changes: a key, or a reference to an
    /// affected e-class.
    affected: HashSet<Value>,
    /// The copy of each affected e-class that has been named so far.
    images: HashMap<Value, Value>,
    container_images: HashMap<Value, Value>,
    /// Memo for [`Walk::container_leaves`].
    container_leaves: HashMap<Value, Vec<(ArcSort, Value)>>,
}

/// Substitute `map` through the sub-e-graph reachable from `root`, returning
/// the root of the copy. See the module docs for the semantics.
///
/// `root` must be an e-class that already has rows: one the query bound, or
/// one from an earlier command. A root the enclosing action just built is not
/// in the tables yet and comes back unchanged, with no error.
pub fn substitute<'db>(
    state: &mut FullState<'_, 'db>,
    root: Value,
    map: &BTreeMap<Value, Value>,
    skip: &std::collections::HashSet<String>,
) -> Result<Value, Error> {
    substitute_with_sort(state, root, map, skip, None)
}

fn substitute_with_sort(
    state: &mut FullState<'_, '_>,
    root: Value,
    map: &BTreeMap<Value, Value>,
    skip: &HashSet<String>,
    root_sort: Option<&str>,
) -> Result<Value, Error> {
    if let Some(target) = map.get(&root) {
        return Ok(*target);
    }
    if map.is_empty() {
        return Ok(root);
    }

    let catalog = state.constructor_catalog()?;
    let mut walk = Walk {
        catalog: &catalog,
        skip,
        map,
        snapshot: Snapshot::default(),
        affected: HashSet::new(),
        images: HashMap::new(),
        container_images: HashMap::new(),
        container_leaves: HashMap::new(),
    };
    let root_sort = root_sort.and_then(|name| {
        catalog
            .by_output
            .get(name)?
            .first()
            .map(|&index| catalog.constructors[index].schema.output.clone())
    });
    walk.collect(state, root, root_sort)?;
    walk.mark();
    if !walk.affected.contains(&root) {
        return Ok(root);
    }
    walk.build(state, root)
}

/// A pending e-class visit in the collect pass. The sort is unknown for the
/// root only in the generic API; typed registrations supply it.
struct Visit(Value, Option<ArcSort>);

impl Walk<'_> {
    /// Gather the reachable e-nodes, container contents, and e-class
    /// dependency edges, starting from `root`.
    fn collect(
        &mut self,
        state: &FullState<'_, '_>,
        root: Value,
        root_sort: Option<ArcSort>,
    ) -> Result<(), Error> {
        let mut stack = vec![Visit(root, root_sort)];
        while let Some(Visit(value, sort)) = stack.pop() {
            if self.snapshot.nodes.contains_key(&value) {
                continue;
            }
            // E-class ids all come from one counter, so probing a constructor
            // of the wrong sort just finds nothing. That is what makes the
            // root's unknown sort affordable: it costs one probe per eq-sorted
            // constructor, once.
            let candidates: Vec<usize> = match &sort {
                Some(sort) => self
                    .catalog
                    .by_output
                    .get(sort.name())
                    .cloned()
                    .unwrap_or_default(),
                None => (0..self.catalog.constructors.len()).collect(),
            };
            let mut nodes = Vec::new();
            for index in candidates {
                let mut rows = Vec::new();
                let ctor = &self.catalog.constructors[index].schema;
                if self.skip.contains(&ctor.name) {
                    continue;
                }
                state.enodes_for_eclass(&ctor.name, value, |enode| {
                    rows.push((enode.children.to_vec(), enode.subsumed));
                })?;
                nodes.extend(rows.into_iter().map(|(children, subsumed)| ENode {
                    ctor: index,
                    children,
                    subsumed,
                }));
            }

            let mut deps = Vec::new();
            for node in &nodes {
                let metadata = &self.catalog.constructors[node.ctor];
                let ctor = &metadata.schema;
                for ((child, child_sort), kind) in node
                    .children
                    .iter()
                    .zip(&ctor.input)
                    .zip(metadata.input_kinds.iter().copied())
                {
                    match kind {
                        Kind::Eclass => {
                            deps.push(*child);
                            stack.push(Visit(*child, Some(child_sort.clone())));
                        }
                        Kind::Container(type_id) => {
                            for (leaf_sort, leaf) in
                                self.container_leaves(state, *child, type_id, child_sort)
                            {
                                deps.push(leaf);
                                stack.push(Visit(leaf, Some(leaf_sort)));
                            }
                        }
                        Kind::Opaque => {}
                    }
                }
            }
            // Recorded after the children are read, but before they are
            // visited, so a cycle back to `value` terminates.
            self.snapshot.nodes.insert(value, nodes);
            self.snapshot.deps.insert(value, deps);
        }
        Ok(())
    }

    /// The e-class leaves of a container value, flattened through nested
    /// containers, recording its contents on the way. Recursion is bounded by
    /// how deeply the program's sorts nest containers.
    fn container_leaves(
        &mut self,
        state: &FullState<'_, '_>,
        value: Value,
        type_id: TypeId,
        sort: &ArcSort,
    ) -> Vec<(ArcSort, Value)> {
        if let Some(leaves) = self.container_leaves.get(&value) {
            return leaves.clone();
        }
        let contents = sort.inner_values(state.container_values(), value);
        let mut leaves = Vec::new();
        for (inner_sort, inner) in &contents {
            match kind_of(inner_sort) {
                Kind::Eclass => leaves.push((inner_sort.clone(), *inner)),
                Kind::Container(inner_type_id) => {
                    leaves.extend(self.container_leaves(state, *inner, inner_type_id, inner_sort))
                }
                Kind::Opaque => {}
            }
        }
        self.snapshot.containers.insert(value, (type_id, contents));
        self.container_leaves.insert(value, leaves.clone());
        leaves
    }

    /// Least fixpoint of "references something the substitution changes",
    /// seeded with the map keys the walk actually reached.
    fn mark(&mut self) {
        let mut users: HashMap<Value, Vec<Value>> = HashMap::new();
        for (owner, deps) in &self.snapshot.deps {
            for dep in deps {
                users.entry(*dep).or_default().push(*owner);
            }
        }

        let mut queue: VecDeque<Value> = self
            .map
            .keys()
            .copied()
            .filter(|key| self.snapshot.nodes.contains_key(key))
            .collect();
        self.affected.extend(queue.iter().copied());
        while let Some(value) = queue.pop_front() {
            for user in users.get(&value).into_iter().flatten() {
                if self.affected.insert(*user) {
                    queue.push_back(*user);
                }
            }
        }
    }

    /// Copy the affected e-classes and return the root's image.
    ///
    /// Every copied e-node goes in through `lookup_or_insert`, so egglog names
    /// the copy's e-class — nothing here invents an id. That is why this is a
    /// sweep rather than a single postorder pass: an e-node can only be copied
    /// once its children have copies, and a cycle in the copied region needs
    /// one e-node whose children all lie outside it to get started. A cycle
    /// with no such e-node is reported instead.
    fn build(&mut self, state: &mut FullState<'_, '_>, root: Value) -> Result<Value, Error> {
        // Children before parents, so the acyclic case finishes in one sweep.
        let order = self.postorder(root);
        let mut pending: Vec<(Value, Vec<ENode>)> = order
            .into_iter()
            .map(|eclass| {
                let nodes = self.snapshot.nodes.remove(&eclass).unwrap_or_default();
                (eclass, nodes)
            })
            .collect();

        loop {
            let mut progress = false;
            for (eclass, nodes) in pending.iter_mut() {
                let mut blocked = Vec::new();
                for node in std::mem::take(nodes) {
                    let Some(args) = self.copied_args(state, &node) else {
                        blocked.push(node);
                        continue;
                    };
                    let name = self.catalog.constructors[node.ctor].schema.name.clone();
                    // A subsumed source row's copy is born retired
                    // (add_subsumed, our egglog fork): a live window —
                    // even one round — re-arms the orbit our :subsume
                    // termination discipline exists to prevent. A COMMITTED
                    // row is never force-retired (independent live material
                    // keeps its flag; an already-subsumed original merges
                    // sticky either way).
                    let copy =
                        if node.subsumed && !state.contains(&name, RawValues(args.clone()))? {
                            state.add_subsumed(&name, RawValues(args))?
                        } else {
                            state.add(&name, RawValues(args))?
                        };
                    match self.images.get(eclass) {
                        // The first e-node copied names the class; the rest are
                        // further ways to say the same one.
                        None => {
                            self.images.insert(*eclass, copy);
                        }
                        Some(image) if *image != copy => state.union(copy, *image)?,
                        Some(_) => {}
                    }
                    progress = true;
                }
                *nodes = blocked;
            }
            if !progress {
                break;
            }
        }

        if let Some((eclass, blocked)) = pending.iter().find(|(_, nodes)| !nodes.is_empty()) {
            // DIAGNOSTIC (vendoring delta): name the cycle's anatomy — every
            // row of the blocked class including the subsumed rows the walk
            // skipped, and which children are still imageless.
            for metadata in &self.catalog.constructors {
                let ctor = &metadata.schema;
                if self.skip.contains(&ctor.name) {
                    continue;
                }
                let mut rows = Vec::new();
                let _ = state.enodes_for_eclass(&ctor.name, *eclass, |enode| {
                    rows.push((enode.children.to_vec(), enode.subsumed));
                });
                for (children, subsumed) in rows {
                    let status: Vec<String> = children
                        .iter()
                        .map(|child| {
                            if self.map.contains_key(child) {
                                format!("{child:?}=KEY")
                            } else if !self.affected.contains(child) {
                                format!("{child:?}=shared")
                            } else if self.images.contains_key(child) {
                                format!("{child:?}=copied")
                            } else {
                                format!("{child:?}=PENDING")
                            }
                        })
                        .collect();
                    eprintln!(
                        "[subst-cycle] {eclass:?} row {}({}){}",
                        ctor.name,
                        status.join(", "),
                        if subsumed { " [SUBSUMED->skipped]" } else { "" }
                    );
                }
            }
            let mut ctors: Vec<&str> = blocked
                .iter()
                .map(|node| self.catalog.constructors[node.ctor].schema.name.as_str())
                .collect();
            ctors.sort_unstable();
            ctors.dedup();
            return Err(error(format!(
                "no order copies e-class {eclass:?}: its remaining e-nodes ({}) all refer to \
                 copies that do not exist yet. Every cycle in the substituted region needs at \
                 least one e-node whose children all lie outside it.",
                ctors.join(", "),
            )));
        }

        self.images
            .get(&root)
            .copied()
            .ok_or_else(|| error(format!("the root e-class {root:?} was not copied")))
    }

    /// The affected e-classes that need copying, children before parents.
    fn postorder(&self, root: Value) -> Vec<Value> {
        enum Frame {
            Enter(Value),
            Exit(Value),
        }

        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![Frame::Enter(root)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(eclass) => {
                    if !seen.insert(eclass) {
                        continue;
                    }
                    stack.push(Frame::Exit(eclass));
                    for dep in self.snapshot.deps.get(&eclass).into_iter().flatten() {
                        if self.needs_copy(*dep) && !seen.contains(dep) {
                            stack.push(Frame::Enter(*dep));
                        }
                    }
                }
                Frame::Exit(eclass) => order.push(eclass),
            }
        }
        order
    }

    /// Whether this e-class gets a copy: affected, and not a key (a key is
    /// replaced outright rather than copied).
    fn needs_copy(&self, eclass: Value) -> bool {
        self.affected.contains(&eclass) && !self.map.contains_key(&eclass)
    }

    /// The substituted children of `node`, or `None` if some child's copy does
    /// not exist yet.
    fn copied_args(&mut self, state: &mut FullState<'_, '_>, node: &ENode) -> Option<Vec<Value>> {
        let metadata = &self.catalog.constructors[node.ctor];
        let ctor = &metadata.schema;
        let mut args = Vec::with_capacity(node.children.len());
        for ((child, _child_sort), kind) in node
            .children
            .iter()
            .zip(&ctor.input)
            .zip(metadata.input_kinds.iter().copied())
        {
            let image = match kind {
                Kind::Eclass => self.eclass_image(*child)?,
                Kind::Container(type_id) => self.container_image(state, *child, type_id)?,
                Kind::Opaque => *child,
            };
            args.push(image);
        }
        Some(args)
    }

    fn eclass_image(&self, eclass: Value) -> Option<Value> {
        if let Some(target) = self.map.get(&eclass) {
            return Some(*target);
        }
        if !self.affected.contains(&eclass) {
            return Some(eclass);
        }
        self.images.get(&eclass).copied()
    }

    /// The interned value of `container` with its contents substituted, or
    /// `container` itself if nothing inside it changed. `None` while any
    /// e-class inside it is still waiting for its copy.
    fn container_image(
        &mut self,
        state: &mut FullState<'_, '_>,
        container: Value,
        type_id: TypeId,
    ) -> Option<Value> {
        if let Some(image) = self.container_images.get(&container) {
            return Some(*image);
        }
        let Some((_, contents)) = self.snapshot.containers.get(&container).cloned() else {
            return Some(container);
        };

        // Nothing is interned until every value inside resolves, so a blocked
        // container leaves no half-substituted copy behind.
        let mut remap: HashMap<Value, Value> = HashMap::new();
        for (inner_sort, inner) in &contents {
            let image = match kind_of(inner_sort) {
                Kind::Eclass => self.eclass_image(*inner)?,
                Kind::Container(inner_type_id) => {
                    self.container_image(state, *inner, inner_type_id)?
                }
                Kind::Opaque => continue,
            };
            if image != *inner {
                remap.insert(*inner, image);
            }
        }

        let image = if remap.is_empty() {
            container
        } else {
            let mapped = state.map_container(type_id, container, &|value| {
                remap.get(&value).copied().unwrap_or(value)
            });
            // `collect` already read this value's contents through the same
            // sort, so it is a container of this type.
            debug_assert!(mapped.is_some(), "{container:?} is not a {type_id:?}");
            mapped.unwrap_or(container)
        };
        self.container_images.insert(container, image);
        Some(image)
    }
}

fn error(message: String) -> Error {
    Error::BackendError(format!("{SUBST}: {message}"))
}

/// Substitute through the sub-e-graph reachable from `root`, returning the
/// e-class of the copy. The top-level form of the [`SUBST`] primitive.
///
/// `map` must be a `Map` container value whose key and value sorts are the same
/// eq-sort; `root` may be of any eq-sort. Constructor rows reachable from
/// `root` are copied with each key e-class replaced by its mapped value;
/// e-classes the substitution does not affect are shared with the original
/// rather than copied.
///
/// `root` must be an e-class that already has rows — see the module docs.
///
/// Errors if the substituted region contains a cycle in which every e-node
/// refers back into the cycle, since naming that copy would require an e-class
/// id no row produces.
pub fn subst(egraph: &mut EGraph, root: Value, map: Value) -> Result<Value, Error> {
    egraph.update(|mut state| {
        let entries = match state.value_to_container::<MapContainer>(map) {
            Some(entries) => entries.data.clone(),
            None => return Err(error(format!("{map:?} is not a Map container value"))),
        };
        substitute(&mut state, root, &entries, &Default::default())
    })
}

/// The `unstable-subst` primitive.
///
/// `skip` (vendoring delta): constructor tables whose rows are MEMO
/// METADATA, not term structure — e.g. our `int-subst-of (IntExpr
/// IndexMap) -> IntExpr`, the union-on-set memo idiom. Substitution
/// distributes over syntax, never over memo tables keyed by syntax:
/// walking a memo row and substituting inside its KEY mints
/// memo rows about the WRONG key (subst_aliased_min: the sigma-1 walk
/// copied `int-subst-of(c_row, row0_map)` out of the zero class as
/// `int-subst-of(1, row0_map)` and welded 0 = 1). Upstream design gap
/// to raise with #60: "constructor rows = the term structure" is
/// false under the memo-constructor idiom.
#[derive(Clone)]
pub struct Subst {
    pub skip: std::collections::HashSet<String>,
}

impl Primitive for Subst {
    fn name(&self) -> &str {
        SUBST
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(SubstTypeConstraint {
            span: span.clone(),
            name: SUBST,
            root_sort: None,
        })
    }
}

impl FullPrim for Subst {
    fn apply<'a, 'db>(&self, state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
        apply_substitution(state, args, &self.skip, None)
    }
}

/// A typed registration of the same substitution semantics. The type checker
/// enforces the root sort; the walk probes only constructors of that sort for
/// the root, then follows each child's declared sort (including containers).
#[derive(Clone)]
pub struct TypedSubst {
    pub name: &'static str,
    pub root_sort: &'static str,
    pub skip: HashSet<String>,
}

impl Primitive for TypedSubst {
    fn name(&self) -> &str {
        self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(SubstTypeConstraint {
            span: span.clone(),
            name: self.name,
            root_sort: Some(self.root_sort),
        })
    }
}

impl FullPrim for TypedSubst {
    fn apply<'a, 'db>(&self, state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
        apply_substitution(state, args, &self.skip, Some(self.root_sort))
    }
}

fn apply_substitution(
    mut state: FullState<'_, '_>,
    args: &[Value],
    skip: &HashSet<String>,
    root_sort: Option<&str>,
) -> Option<Value> {
    let [root, map] = args else { return None };
    // Release the container registry borrow before interning new containers.
    let entries = state.value_to_container::<MapContainer>(*map)?.data.clone();
    match substitute_with_sort(&mut state, *root, &entries, skip, root_sort) {
        Ok(image) => Some(image),
        Err(err) => {
            // FullPrim cannot return an Error; preserve the loud reason and
            // primitive panic of the generic registration.
            eprintln!("[unstable-subst] {err}");
            state.panic();
            None
        }
    }
}

/// `(unstable-subst root map) : (R, Map<K, K>) -> R` for any eq-sort `R`.
///
/// `R` is free rather than pinned to `K` because a substitution reaches through
/// every sort in the term structure, so a root of one sort can perfectly well
/// be rewritten by a map over another. `K` must be an eq-sort mapping to
/// itself: replacing a `K`-sorted child with a value of another sort would
/// produce an ill-typed row.
struct SubstTypeConstraint {
    span: Span,
    name: &'static str,
    root_sort: Option<&'static str>,
}

impl TypeConstraint for SubstTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        let [root, map, out] = arguments else {
            return vec![constraint::impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: self.span.clone(),
                        head: self.name.to_owned(),
                        args: arguments.to_vec(),
                    },
                    expected: 3,
                },
            )];
        };

        let mut cs: Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> =
            vec![constraint::eq(root.clone(), out.clone())];

        // One instantiation per declared sort that could stand in each
        // position; `xor` defers until the surrounding program pins it down.
        //
        // A `Map` sort is identified by the Rust type its values intern
        // under, since the `ContainerSort` impl behind an `ArcSort` is
        // wrapped in a private type that out-of-tree code cannot downcast to.
        let mut map_sorts: Vec<ArcSort> = typeinfo.get_arcsorts_by(|sort| {
            sort.value_type() == Some(TypeId::of::<MapContainer>())
                && match sort.inner_sorts().as_slice() {
                    [key, value] => key.is_eq_sort() && key.name() == value.name(),
                    _ => false,
                }
        });
        map_sorts.sort_by_key(|sort| sort.name().to_owned());
        cs.push(constraint::xor(
            map_sorts
                .into_iter()
                .map(|sort| constraint::assign(map.clone(), sort))
                .collect(),
        ));

        let mut eq_sorts = typeinfo.get_arcsorts_by(|sort| {
            sort.is_eq_sort() && self.root_sort.is_none_or(|name| sort.name() == name)
        });
        eq_sorts.sort_by_key(|sort| sort.name().to_owned());
        cs.push(constraint::xor(
            eq_sorts
                .into_iter()
                .map(|sort| {
                    constraint::and(vec![
                        constraint::assign(root.clone(), sort.clone()),
                        constraint::assign(out.clone(), sort),
                    ])
                })
                .collect(),
        ));

        cs
    }
}

#[cfg(test)]
mod metadata_cache_tests {
    use super::*;

    fn graph() -> EGraph {
        let mut egraph = EGraph::default();
        egraph.add_full_primitive(
            Subst {
                skip: ["Memo".to_owned()].into_iter().collect(),
            },
            None,
        );
        egraph
    }
    const BASE: &str = r#"
        (datatype Node (Leaf i64) (Pair Node Node))
        (sort Mapping (Map Node Node))
        (let a (Leaf 0))
        (let b (Leaf 1))
        (let sigma (map-insert (map-empty) a b))
    "#;

    #[test]
    fn typed_root_matches_generic_across_sorts_containers_and_unions() {
        let mut egraph = graph();
        egraph.add_full_primitive(
            TypedSubst {
                name: "subst-node",
                root_sort: "Node",
                skip: ["Memo".to_owned()].into_iter().collect(),
            },
            None,
        );
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    r#"{BASE}
            (datatype Other (Cross Node))
            (sort Others (Vec Other))
            (constructor Held (Others) Node)
            (let p (Held (vec-of (Cross (Pair a a)))))
            (let generic (unstable-subst p sigma))
            (let typed (subst-node p sigma))
            (check (= generic typed))
            (check (= typed (Held (vec-of (Cross (Pair b b))))))
            (constructor Wrap (Node) Node)
            (let wrapped (Wrap p))
            (union wrapped p)
            (let generic2 (unstable-subst p sigma))
            (let typed2 (subst-node p sigma))
            (check (= generic2 typed2))
            (let empty (subst-node p (map-empty)))
            (check (= empty p))
            (let key (subst-node a sigma))
            (check (= key b))
        "#
                ),
            )
            .unwrap();
        assert!(
            egraph
                .parse_and_run_program(None, "(let wrong (subst-node (Cross a) sigma))")
                .is_err(),
            "the typed registration must reject roots of another sort"
        );
    }

    #[test]
    fn cached_walk_observes_late_constructors_and_new_rows() {
        let mut egraph = graph();
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    r#"{BASE}
            (let p (Pair a a))
            (let copied (unstable-subst p sigma))
            (check (= copied (Pair b b)))
            (constructor Wrap (Node) Node)
            (let w (Wrap p))
            (let wrapped-copy (unstable-subst w sigma))
            (check (= wrapped-copy (Wrap (Pair b b))))
            (let fresh (Pair (Leaf 2) a))
            (let fresh-copy (unstable-subst fresh sigma))
            (check (= fresh-copy (Pair (Leaf 2) b)))
        "#
                ),
            )
            .unwrap();
    }

    #[test]
    fn cached_walk_preserves_container_capture_and_memo_exclusion() {
        let mut egraph = graph();
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    r#"{BASE}
            (sort Nodes (Vec Node))
            (constructor Pack (Nodes) Node)
            (constructor Memo (Node Mapping) Node)
            (let untouched (Leaf 2))
            (union (Memo untouched sigma) untouched)
            (let p (Pair untouched a))
            (let copied (unstable-subst p sigma))
            (check (= copied (Pair untouched b)))
            (let packed (Pack (vec-of a untouched)))
            (let packed-copy (unstable-subst packed sigma))
            (check (= packed-copy (Pack (vec-of b untouched))))
            (sort Thunk (UnstableFn () Node))
            (constructor Held (Thunk) Node)
            (let held (Held (unstable-fn "Pair" a a)))
            (let held-copy (unstable-subst held sigma))
            (check (= held-copy (Held (unstable-fn "Pair" b b))))
            (let empty-copy (unstable-subst p (map-empty)))
            (check (= empty-copy p))
        "#
                ),
            )
            .unwrap();
    }
    #[test]
    fn cached_walk_keeps_subsumed_rows_and_handles_grounded_cycles() {
        let mut egraph = graph();
        egraph
            .parse_and_run_program(None, "(datatype Node (Leaf i64) (Pair Node Node))")
            .unwrap();
        let (a, b, c, p) = egraph
            .update(|mut state| {
                let a = state.add("Leaf", (0_i64,))?;
                let b = state.add("Leaf", (1_i64,))?;
                let c = state.add("Leaf", (2_i64,))?;
                let p = state.add("Pair", (a, c))?;
                Ok((a, b, c, p))
            })
            .unwrap();
        // Populate the schema cache before changing the live row's fence.
        let _ = egraph.read(|s| s.constructor_catalog()).unwrap();
        egraph
            .update(|mut state| state.subsume("Pair", (a, c)))
            .unwrap();
        let sigma = BTreeMap::from([(a, b)]);
        let copy = egraph
            .update(|mut state| substitute(&mut state, p, &sigma, &Default::default()))
            .unwrap();
        egraph.read(|state| {
            let mut rows = 0;
            state
                .enodes_for_eclass("Pair", copy, |node| {
                    assert_eq!(node.children, &[b, c]);
                    assert!(
                        node.subsumed,
                        "a cached schema must not resurrect a retired spelling"
                    );
                    rows += 1;
                })
                .unwrap();
            assert_eq!(rows, 1);
        });
        egraph
            .update(|mut state| {
                let cycle = state.add("Pair", (c, a))?;
                state.union(c, cycle)
            })
            .unwrap();
        egraph
            .update(|mut state| substitute(&mut state, c, &sigma, &Default::default()))
            .unwrap();
    }
}
