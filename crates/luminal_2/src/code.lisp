; -------- SYMBOLIC ALGEBRA -------
(ruleset expr)
(datatype Expression
	(MNum i64)
	(MVar String)
	(MAdd Expression Expression)
	(MSub Expression Expression)
	(MMul Expression Expression)
	(MDiv Expression Expression)
	(MMod Expression Expression)
	(MMin Expression Expression)
	(MMax Expression Expression)
	(MAnd Expression Expression)
	(MOr Expression Expression)
	(MGte Expression Expression)
	(MLt Expression Expression)
	(MFloorTo Expression Expression)
    (MReplace Expression Expression Expression)
    (MAccum String) ; this marks that we feed the output (also marked with MAccum) back in
)

; Associative
(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)) :ruleset expr)
(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)) :ruleset expr)

; Constant folding
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)) :ruleset expr)
(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)) :ruleset expr)
(rewrite (MMul (MNum a) (MNum b)) (MNum (* a b)) :when ((< a 10000) (< b 10000)) :ruleset expr)
(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))) :ruleset expr)
(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)) :ruleset expr)
(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)) :ruleset expr)
(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)) :ruleset expr)

; Simple reductions
(rewrite (MAdd a (MNum 0)) a :ruleset expr)
(rewrite (MMul a (MNum 1)) a :ruleset expr)
(rewrite (MMul a (MNum 0)) (MNum 0) :ruleset expr)
(rewrite (MDiv a (MNum 1)) a :ruleset expr)
(rewrite (MMul (MDiv ?a ?b) ?b) (MFloorTo ?a ?b) :ruleset expr)
(rewrite (MAdd (MFloorTo ?a ?b) (MMod ?a ?b)) ?a :ruleset expr)
;(rewrite (MDiv ?a ?a) (MNum 1) :ruleset expr) ; why does this cause kernels to incorrectly oversimplify?
;(rewrite (MDiv (MMul ?x ?y) ?y) ?x :ruleset expr) ; and this?
(rewrite (MMod (MMul ?x ?y) ?y) (MNum 0) :ruleset expr)
(rewrite (MDiv (MMul ?x ?y) ?z) (MMul ?x (MDiv ?y ?z)) :ruleset expr)
(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?y)) :when ((>= ?z ?y) (= 0 (% ?y ?z))) :ruleset expr) ; nested mods
(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?z)) :when ((>= ?y ?z) (= 0 (% ?z ?y))) :ruleset expr)

; reduce contiguous multidimensional indexing
(rewrite
	(MAdd
		(MMul (MNum ?outerStride) (MMod (MDiv (MVar "z") (MNum ?innerSize)) (MNum ?outerSize)))
		(MMul (MNum ?innerStride) (MMod (MVar "z") (MNum ?innerSize)))
	)
	(MMul (MNum ?innerStride) (MMod (MVar "z") (MNum (* ?innerSize ?outerSize))))
	:ruleset expr
)


; Replacement
(rewrite (MReplace ?x ?y ?z) ?z :when ((= ?x ?y)) :ruleset expr)
(rewrite (MReplace (MAdd ?a ?b) ?x ?y) (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MSub ?a ?b) ?x ?y) (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMul ?a ?b) ?x ?y) (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MDiv ?a ?b) ?x ?y) (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMod ?a ?b) ?x ?y) (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMin ?a ?b) ?x ?y) (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMax ?a ?b) ?x ?y) (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MFloorTo ?a ?b) ?x ?y) (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
; leave numbers unchanged
(rewrite (MReplace (MNum ?n) ?x ?y) (MNum ?n) :ruleset expr)
(rewrite (MReplace (MAccum ?acc) ?x ?y) (MAccum ?acc) :ruleset expr)

; leave other vars unchanged
(rewrite (MReplace (MVar ?v) (MVar ?x) ?y) (MVar ?v) :when ((!= ?v ?x)) :ruleset expr)

; reduce multi-dim squeezed indexing into simple multiplicative indexing
(rewrite
  (MAdd (MMul (MNum (* d n2)) (MMod (MDiv ?v (MNum d)) (MNum m)))
        (MMul (MNum n2) (MMod ?v (MNum d))))
  (MMul ?v (MNum n2))
	:ruleset expr
)

; -------- IR --------
(ruleset ir)
(ruleset ir-prop)
(ruleset ir-generic)
(datatype LoopType (Loop String Expression))
(datatype*
 	(IR
  		; General kernel stuff
     	(GMEM String)
     	(LoopIn IR LoopType Expression)
     	(LoopOut IR LoopType Expression)
      	(SMEM)
       	(SMEMLoad IR IR)
        (SMEMRead IR IR)

        ; Unary Ops
     	(Exp2 IR)
      	(Log2 IR)
    	(Sqrt IR)
     	(Sin IR)
      	(Recip IR)
       	(Neg IR)

        ; Binary Ops
     	(Add IR IR)
     	(Mul IR IR)
      	(Max IR IR)

        ; search helpers
        (Unary String IR)
     	(Binary String IR IR)

      	; propogation patterns
      	(SwapLoops IR String String) ; Swap two loops, identified by their strings
       	(TileLoop IR String) ; Tile a loop, identified by it's string
        (UnpadLoop IR String) ; Remove a padding loop, identified by it's string
        (MergeLoops IR String String) ; Merge loops, identified by their strings
        (Fusable IR) ; Dictates that this IR is fuseable with downstream loopin -> loopout

    	; propogation pattern helpers
     	(PropOneArg String IR String) ; Generic prop one arg back
     	(PropTwoArgs String IR String String) ; Generic prop two args back
     )
)

; -------------- HELPERS ---------------

; Convert to and from generic unary ops
(birewrite (Exp2 ?x) (Unary "Exp2" ?x) :ruleset ir-generic)
(birewrite (Log2 ?x) (Unary "Log2" ?x) :ruleset ir-generic)
(birewrite (Sqrt ?x) (Unary "Sqrt" ?x) :ruleset ir-generic)
(birewrite (Sin ?x) (Unary "Sin" ?x) :ruleset ir-generic)
(birewrite (Recip ?x) (Unary "Recip" ?x) :ruleset ir-generic)
(birewrite (Neg ?x) (Unary "Neg" ?x) :ruleset ir-generic)
(birewrite (Add ?a ?b) (Binary "Add" ?a ?b) :ruleset ir-generic)
(birewrite (Mul ?a ?b) (Binary "Mul" ?a ?b) :ruleset ir-generic)
(birewrite (Max ?a ?b) (Binary "Max" ?a ?b) :ruleset ir-generic)

; Communative binary ops
;(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a) :ruleset ir)
; distributive/associative skeletons so sums and products re-associate
;(rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)) :ruleset ir)
;(rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)) :ruleset ir)

; ---------- RULES ----------

; remove pad loop
(rewrite
 	(LoopOut (Unary ?un (LoopIn ?x (Loop ?loop (MNum 1)) (MNum 0))) (Loop ?loop (MNum 1)) (MNum 0))
	(Unary ?un ?x)
	 :ruleset ir
)
(rewrite
 	(LoopOut (Binary ?bin (LoopIn ?a (Loop ?loop (MNum 1)) (MNum 0)) (LoopIn ?b (Loop ?loop (MNum 1)) (MNum  0))) (Loop ?loop (MNum 1)) (MNum 0))
	(Binary ?bin ?a ?b)
	 :ruleset ir
)
; add pad loop
(rewrite
	(LoopOut (Unary ?un ?x) (Loop ?l ?r) ?s)
	(LoopOut (LoopOut (Unary ?un (LoopIn ?x (Loop "newpad" (MNum 1)) (MNum 0))) (Loop "newpad" (MNum 1)) (MNum 0)) (Loop ?l ?r) ?s)
	:when ((!= ?r (MNum 1)) (!= ?s (MNum 0)))
	 :ruleset ir
)
(rewrite
	(LoopOut (Binary ?bin ?a ?b) (Loop ?l ?r) ?s)
	(LoopOut (LoopOut (Binary ?bin (LoopIn ?a (Loop "newpad" (MNum 1)) (MNum 0)) (LoopIn ?b (Loop "newpad" (MNum 1)) (MNum 0))) (Loop "newpad" (MNum 1)) (MNum 0)) (Loop ?l ?r) ?s)
	:when ((!= ?r (MNum 1)) (!= ?s (MNum 0)))
	 :ruleset ir
)
; remove unnessecary modulo
(rewrite (LoopIn ?x (Loop ?l ?range) (MMul ?st (MMod (MVar "z") ?range))) (LoopIn ?x (Loop ?l ?range) (MMul ?st (MVar "z"))) :ruleset expr)
(rewrite (LoopOut ?x (Loop ?l ?range) (MMul ?st (MMod (MVar "z") ?range))) (LoopOut ?x (Loop ?l ?range) (MMul ?st (MVar "z"))) :ruleset expr)

; Loop Fusion
(rewrite (LoopIn (LoopOut ?x (Loop ?lo ?range) ?st) (Loop ?li ?range) ?st) ?x :ruleset ir)

; Specialized swap loops
(rewrite
	(LoopOut
		(LoopOut
			(Binary ?bin
				(LoopIn
					(LoopIn ?a (Loop ?outL ?out) ?outASt)
					(Loop ?inL ?in)
					?inASt
				)
				(LoopIn
					(LoopIn ?b (Loop ?outL ?out) ?outBSt)
					(Loop ?inL ?in)
					?inBSt
				)
			)
			(Loop ?inL ?in)
			?inSt
		)
		(Loop ?outL ?out)
		?outSt
	)
	(LoopOut
		(LoopOut
			(Binary ?bin
				(LoopIn
					(LoopIn ?a (Loop ?inL ?in) ?inASt)
					(Loop ?outL ?out)
					?outASt
				)
				(LoopIn
					(LoopIn ?b (Loop ?inL ?in) ?inBSt)
					(Loop ?outL ?out)
					?outBSt
				)
			)
			(Loop ?outL ?out)
			?outSt
		)
		(Loop ?inL ?in)
		?inSt
	)
	 :ruleset ir
)

; Tiling
(let tileFactor 2)
(rewrite
	(LoopOut ?body (Loop ?loop (MNum ?range)) ?stride)
	(LoopOut
		(LoopOut
			(TileLoop ?body ?loop)
			(Loop (+ ?loop "_tile") (MNum tileFactor))
			?stride
		)
		(Loop (+ ?loop "_out") (MNum (/ ?range tileFactor)))
		(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
	)
	:when ((> ?range tileFactor) (= (% ?range tileFactor) 0))
	 :ruleset ir
)
(rewrite
	(TileLoop (LoopIn ?body (Loop ?loop (MNum ?range)) ?stride) ?loop)
	(LoopIn
		(LoopIn ?body
			(Loop (+ ?loop "_out") (MNum (/ ?range tileFactor)))
			(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
		)
		(Loop (+ ?loop "_tile") (MNum tileFactor))
		?stride
	)
	 :ruleset ir-prop
)
; propogation
(rewrite
	(TileLoop (LoopIn ?body (Loop ?other ?range) ?stride) ?loop)
	(LoopIn (TileLoop ?body ?loop) (Loop ?other ?range) ?stride)
	:when ((!= ?loop ?other))
	 :ruleset ir-prop
)
(rewrite
	(TileLoop (LoopOut ?body (Loop ?other ?range) ?stride) ?loop)
	(LoopOut (TileLoop ?body ?loop) (Loop ?other ?range) ?stride)
	 :ruleset ir-prop
)
(rewrite
	(TileLoop (Unary ?un ?body) ?loop)
	(Unary ?un (TileLoop ?body ?loop))
	 :ruleset ir-prop
)
(rewrite
	(TileLoop (Binary ?bin ?bodyA ?bodyB) ?loop)
	(Binary ?bin (TileLoop ?bodyA ?loop) (TileLoop ?bodyB ?loop))
	 :ruleset ir-prop
)

; Loop merging
(rewrite
	(LoopOut
		(LoopOut ?x
			(Loop ?i ?rangeI) ?stI
		)
		(Loop ?o ?rangeO) ?stO
	)
	(LoopOut (MergeLoops ?x ?o ?i)
		(Loop (+ ?o ?i) (MMul ?rangeO ?rangeI))
		(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") ?rangeI)) (MReplace ?stI (MVar "z") (MMod (MVar "z") ?rangeI)))
	)
	 :ruleset ir
)
(rewrite
	(MergeLoops
		(LoopIn
			(LoopIn
				?x
				(Loop ?o ?rangeO) ?stO
			)
			(Loop ?i ?rangeI) ?stI
		)
		?o ?i
	)
	(LoopIn
		?x
		(Loop (+ ?o ?i) (MMul ?rangeO ?rangeI))
		(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") ?rangeI)) (MReplace ?stI (MVar "z") (MMod (MVar "z") ?rangeI)))
	)
	 :ruleset ir-prop
)
; propogation
(rewrite
	(MergeLoops (LoopIn ?body (Loop ?other ?range) ?stride) ?o ?i)
	(LoopIn (MergeLoops ?body ?o ?i) (Loop ?other ?range) ?stride)
	:when ((!= ?i ?other))
	 :ruleset ir-prop
)
(rewrite
	(MergeLoops (LoopOut ?body (Loop ?other ?range) ?stride) ?o ?i)
	(LoopOut (MergeLoops ?body ?o ?i) (Loop ?other ?range) ?stride)
	 :ruleset ir-prop
)
(rewrite
	(MergeLoops (Unary ?un ?body) ?o ?i)
	(Unary ?un (MergeLoops ?body ?o ?i))
	 :ruleset ir-prop
)
(rewrite
	(MergeLoops (Binary ?bin ?bodyA ?bodyB) ?o ?i)
	(Binary ?bin (MergeLoops ?bodyA ?o ?i) (MergeLoops ?bodyB ?o ?i))
	 :ruleset ir-prop
)

{code}
(run-schedule
	(saturate ir-generic)
	(repeat {iters}
		(run ir) ; run ir rules once
		(repeat 4 ir-prop)
		(repeat 5 expr)
	)
	(saturate ir-generic) ; why is this needed?
)

(let a0 (GMEM "acc_0"))
(let a1 (LoopIn a0 (Loop "" (MNum 64)) (MNum 0)))
(let a2 (LoopIn a1 (Loop "" (MNum 1)) (MNum 0)))
(let a3 (LoopIn a2 (Loop "" (MNum 1)) (MNum 0)))
(let a4 (LoopIn a3 (Loop "" (MNum 8)) (MAccum "a")))
(let a5 (GMEM "A Load"))
;(let a6 (TileLoop (LoopIn a5 (Loop "" (MNum 512)) (MAdd (MMul (MNum 8) (MMod (MDiv (MVar "z") (MNum 64)) (MNum 8))) (MMod (MVar "z") (MNum 8)))) "0"))
(let a6 (LoopIn a5 (Loop "" (MNum 128)) (MAdd (MMul (MNum 8) (MMod (MDiv (MMul (MMul (MVar "z") (MNum 2)) (MNum 2)) (MNum 64)) (MNum 8))) (MMod (MMul (MMul (MVar "z") (MNum 2)) (MNum 2)) (MNum 8)))))
;(let a6tileouter (LoopIn a6 (Loop "" (MNum 2)) (MAdd (MMul (MNum 8) (MMod (MDiv (MMul (MVar "z") (MNum 2)) (MNum 64)) (MNum 8))) (MMod (MMul (MVar "z") (MNum 2)) (MNum 8)))))
(let a6tile (LoopIn a6 (Loop "" (MNum 4))
	(MAdd
		(MAdd (MMul (MNum 8) (MMod (MDiv (MMul (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 64)) (MNum 8))) (MMod (MMul (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 8)))
		(MAdd (MMul (MNum 8) (MMod (MDiv (MMod (MVar "z") (MNum 2)) (MNum 64)) (MNum 8))) (MMod (MMod (MVar "z") (MNum 2)) (MNum 8)))
	)
))
(let a7 (GMEM "B Load"))
;(let a8 (LoopIn a7 (Loop "0" (MNum 512)) (MAdd (MMod (MDiv (MVar "z") (MNum 8)) (MNum 8)) (MMul (MNum 8) (MMod (MVar "z") (MNum 8))))))
(let a8 (LoopIn a7 (Loop "" (MNum 128)) (MAdd (MMod (MDiv (MMul (MMul (MVar "z") (MNum 2)) (MNum 2)) (MNum 8)) (MNum 8)) (MMul (MNum 8) (MMod (MMul (MMul (MVar "z") (MNum 2)) (MNum 2)) (MNum 8))))))
;(let a8tileouter (LoopIn a8 (Loop "" (MNum 2)) (MAdd (MMod (MDiv (MMul (MVar "z") (MNum 2)) (MNum 8)) (MNum 8)) (MMul (MNum 8) (MMod (MMul (MVar "z") (MNum 2)) (MNum 8))))))
(let a8tile (LoopIn a8 (Loop "" (MNum 4))
	(MAdd
		(MAdd (MMod (MDiv (MMul (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 8)) (MNum 8)) (MMul (MNum 8) (MMod (MMul (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 8))))
		(MAdd (MMod (MDiv (MMod (MVar "z") (MNum 2)) (MNum 8)) (MNum 8)) (MMul (MNum 8) (MMod (MVar "z") (MNum 2))))
	)
))
(let a9 (Mul a6tile a8tile))
(let a10tile (LoopOut a9 (Loop "" (MNum 4)) (MVar "z")))
(let a10 (LoopOut a10tile (Loop "" (MNum 128)) (MMul (MVar "z") (MNum 4))))
(let a11 (LoopIn a10 (Loop "" (MNum 64)) (MMul (MNum 8) (MVar "z"))))
(let a12 (LoopIn a11 (Loop "" (MNum 1)) (MNum 0)))
(let a13 (LoopIn a12 (Loop "" (MNum 1)) (MNum 0)))
(let a14 (LoopIn a13 (Loop "" (MNum 8)) (MVar "z")))
(let a15 (Add a4 a14))
(let a16 (LoopOut a15 (Loop "" (MNum 8)) (MAccum "a")))
(let a17 (LoopOut a16 (Loop "" (MNum 1)) (MVar "z")))
(let a18 (LoopOut a17 (Loop "" (MNum 1)) (MVar "z")))
(let a19 (LoopOut a18 (Loop "" (MNum 64)) (MVar "z")))

(ruleset a-rule)
(rewrite (Loop ?x ?r) (Loop "" ?r) :ruleset a-rule)
(rewrite (TileLoop ?x ?l) (TileLoop ?x "") :ruleset a-rule)
(rewrite (MergeLoops ?x ?l ?o) (MergeLoops ?x "" "") :ruleset a-rule)
(run-schedule (saturate a-rule))

(check (= a19 t19))