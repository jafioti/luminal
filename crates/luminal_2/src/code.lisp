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

; Communative
(rewrite (MAdd a b) (MAdd b a) :ruleset expr)
(rewrite (MMul a b) (MMul b a) :ruleset expr)

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
      	(SwapLoops IR LoopType LoopType) ; Swap two loops, identified by their strings
       	(TileLoop IR String) ; Tile a loop, identified by it's string
        (UnpadLoop IR String) ; Remove a padding loop, identified by it's string
        (MergeLoops IR String String) ; Merge loops, identified by their strings
        (FusedLoops IR Expression) ; Says that we have previously fused a loopout -> loopin here

    	; propogation pattern helpers
     	(PropOneArg String IR String) ; Generic prop one arg back
     	(PropTwoArgs String IR String String) ; Generic prop two args back

      	; tensor core stuff
      	(TCMatmul IR IR Expression Expression Expression Expression Expression Expression) ; input A, input B, A k stride, B k stride, A inner stride, B inner stride, C inner stride, number of K tile loops
       	(TiledMatmulInputA IR i64 Expression)
        (TiledMatmulInputB IR i64 Expression)
        (TiledMatmulAcc IR)
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
(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a) :ruleset ir)
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


; Loop Fusion
(ruleset fusion)
(rewrite (LoopIn (LoopOut (Binary ?bin ?a ?b) (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st) (Binary ?bin ?a ?b) :ruleset fusion)
(rewrite (LoopIn (LoopOut ?a (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st) (FusedLoops ?a ?range) :ruleset fusion)
(rewrite (LoopIn (FusedLoops (LoopOut ?a (Loop ?loopA ?range) ?st) ?fused_range) (Loop ?loopB ?range) ?st) (FusedLoops ?a (MMul ?range ?fused_range)) :ruleset fusion)
(rewrite (LoopIn (FusedLoops (LoopOut (Binary ?bin ?a ?b) (Loop ?loopA ?range) ?st) ?fused_range) (Loop ?loopB ?range) ?st) (Binary ?bin ?a ?b) :ruleset fusion)

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
(let tileFactor 8)
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
;(rewrite
;	(LoopOut
;		(LoopOut ?x
;			(Loop ?i ?rangeI) ?stI
;		)
;		(Loop ?o ?rangeO) ?stO
;	)
;	(LoopOut (MergeLoops ?x ?o ?i)
;		(Loop (+ ?o ?i) (MMul ?rangeO ?rangeI))
;		(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") ?rangeI)) (MReplace ?stI (MVar "z") (MMod (MVar "z") ?rangeI)))
;	)
;	 :ruleset ir
;)
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

; TensorCore
(ruleset tc)
(rewrite
	(LoopIn ; k
		(LoopIn ; pad2
			(LoopIn ; pad1
				(LoopIn ; n
					(LoopIn ; m
						?a
						(Loop ?loop_a_mtile (MNum ?m))
						(MMul (MVar "z") (MNum ?k))
					)
					(Loop ?loop_a_ntile (MNum ?n))
					(MNum 0)
				)
				(Loop ?pad1 (MNum 1))
				(MNum 0)
			)
			(Loop ?pad2 (MNum 1))
			(MNum 0)
		)
		(Loop ?loop_a_kouter (MNum ?k))
		(MVar "z")
	)
	(TiledMatmulInputA ?a ?k (MNum (/ ?k 8)))
	:when ((= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopIn ; k
		(LoopIn ; pad2
			(LoopIn ; pad1
				(LoopIn ; n
					(LoopIn ; m
						?b
						(Loop ?loop_b_mtile (MNum ?m))
						(MNum 0)
					)
					(Loop ?loop_b_ntile (MNum ?n))
					(MVar "z")
				)
				(Loop ?pad1 (MNum 1))
				(MNum 0)
			)
			(Loop ?pad2 (MNum 1))
			(MNum 0)
		)
		(Loop ?loop_b_kouter (MNum ?k))
		(MMul (MVar "z") (MNum ?n))
	)
	(TiledMatmulInputB ?b ?n (MNum (/ ?k 8)))
	:when ((= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopIn ; k outer
		(LoopIn ; pad2
			(LoopIn ; pad1
				(LoopIn ; n tile
					(LoopIn ; m tile
						?acc
						(Loop ?loop_acc_mtile (MNum ?m))
						(MNum 0)
					)
					(Loop ?loop_acc_ntile (MNum ?n))
					(MNum 0)
				)
				(Loop ?pad1 (MNum 1))
				(MNum 0)
			)
			(Loop ?pad2 (MNum 1))
			(MNum 0)
		)
		(Loop ?loop_acc_kouter (MNum ?k))
		(MAccum ?accum)
	)
	(TiledMatmulAcc ?acc)
	:when ((= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopOut ; m
		(LoopOut ; n
			(LoopOut ; pad1
				(LoopOut ; pad2
					 (LoopOut ; k
						(Add
							(Mul
								(TiledMatmulInputA ?a ?k ?k_loops)
								(TiledMatmulInputB ?b ?n ?k_loops)
							)
							; accumulator
							(TiledMatmulAcc ?acc)
						)
						(Loop ?loop_out_k (MNum ?k))
						(MAccum ?acc_outer)
					)
					(Loop ?pad2 (MNum 1))
					(MVar "z")
				)
				(Loop ?pad1 (MNum 1))
				(MVar "z")
			)
			(Loop ?loop_out_n (MNum ?n))
			(MVar "z")
		)
		(Loop ?loop_out_m (MNum ?m))
		(MMul (MVar "z") (MNum ?n))
	)
	(LoopOut ; m outer
		(LoopOut ; n outer
			(LoopOut ; m tile
				(LoopOut ; n tile
					(TCMatmul
						; a
						(LoopIn ; n tile
							(LoopIn ; m tile
								(LoopIn ; n outer
									(LoopIn ; m outer
										?a
										(Loop ?loop_out_m (MNum (/ ?m 8)))
										(MMul (MVar "z") (MNum (* ?k 8)))
									)
									(Loop ?loop_out_n (MNum (/ ?n 8)))
									(MNum 0)
								)
								(Loop (+ ?loop_out_m "_tile") (MNum 8))
								(MNum 0)
							)
							(Loop (+ ?loop_out_n "_tile") (MNum 4))  ; each thread in the matmul does 2 elements
							(MNum 0)
						)
						; b
						(LoopIn ; n tile
							(LoopIn ; m tile
								(LoopIn ; n outer
									(LoopIn ; m outer
										?b
										(Loop ?loop_out_m (MNum (/ ?m 8)))
										(MNum 0)
									)
									(Loop ?loop_out_n (MNum (/ ?n 8)))
									(MMul (MVar "z") (MNum 8))
								)
								(Loop (+ ?loop_out_m "_tile") (MNum 8))
								(MNum 0)
							)
							(Loop (+ ?loop_out_n "_tile") (MNum 4))  ; each thread in the matmul does 2 elements
							(MNum 0)
						)
						; a k stride
						(MMul (MVar "z") (MNum 8))
						; b k stride
						(MMul (MVar "z") (MNum (* ?n 8)))
						; a row size
						(MNum ?k)
						; b row size
						(MNum ?n)
						; c row size
						(MNum ?n)
						; k loops
						?k_loops
					)
					(Loop (+ ?loop_out_n "_tile") (MNum 4))
					(MNum 0)
				)
				(Loop (+ ?loop_out_m "_tile") (MNum 8))
				(MNum 0)
			)
			(Loop ?loop_out_n (MNum (/ ?n 8)))
			(MMul (MVar "z") (MNum 8))
		)
		(Loop ?loop_out_m (MNum (/ ?m 8)))
		(MMul (MVar "z") (MNum (* ?n 8)))
	)
	:ruleset tc
)

; Swap loops
(ruleset swap)
(rewrite
	(LoopOut (LoopOut ?x ?innerLoop ?innerStride) ?outerLoop ?outerStride)
	(LoopOut (LoopOut (SwapLoops ?x ?innerLoop ?outerLoop) ?outerLoop ?outerStride) ?innerLoop ?innerStride)
	:ruleset swap
)
(rewrite
	(SwapLoops (LoopIn (LoopIn ?x ?outerLoop ?outerStride) ?innerLoop ?innerStride) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn ?x ?innerLoop ?innerStride) ?outerLoop ?outerStride)
	:ruleset ir-prop
)
; propogate
(rewrite
	(SwapLoops (LoopOut ?x ?loop ?stride) ?innerLoop ?outerLoop)
	(LoopOut (SwapLoops ?x ?innerLoop ?outerLoop) ?loop ?stride)
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (LoopIn ?x ?loop ?stride) ?innerLoop ?outerLoop)
	(LoopIn (SwapLoops ?x ?innerLoop ?outerLoop) ?loop ?stride)
	:when ((!= ?loop ?outerLoop))
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (Binary ?bin ?a ?b) ?innerLoop ?outerLoop)
	(Binary ?bin (SwapLoops ?a ?innerLoop ?outerLoop) (SwapLoops ?b ?innerLoop ?outerLoop))
	:ruleset ir-prop
)

{code}
(run-schedule
	(run ir-generic)
	(repeat 5
		(run ir)
		(run ir-prop)
		(run expr)
		(repeat 2 fusion)
		(run ir-generic)
	)
	;(run ir) ; run ir rules once
	;(run swap) ; run swap every other run
	;(repeat 5 ir-prop)
	;(repeat 3 expr)
	;(run swap)
	;(repeat 3 ir-prop)
	(run ir-generic)
	(repeat 3 tc)
)
