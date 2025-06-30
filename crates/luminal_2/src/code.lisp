(datatype Math
	(MNum i64)
	(MVar String)
	(MAdd Math Math)
	(MSub Math Math)
	(MMul Math Math)
	(MDiv Math Math)
	(MMod Math Math)
	(MMin Math Math)
	(MMax Math Math)
	(MAnd Math Math)
	(MOr Math Math)
	(MGte Math Math)
	(MLt Math Math)
	(MFloorTo Math Math)
    (MReplace Math Math Math)
    (MAccum String) ; this marks that we feed the output (also marked with MAccum) back in
)

; Communative
;(rewrite (MAdd a b) (MAdd b a))
;(rewrite (MMul a b) (MMul b a))
;(rewrite (MMin a b) (MMin b a))
;(rewrite (MMax a b) (MMax b a))
;(rewrite (MAnd a b) (MAnd b a))
;(rewrite (MOr a b) (MOr b a))

; Associative
(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)))
(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)))
;(rewrite (MDiv (MDiv a b) c) (MDiv a (MMul b c)))
;(rewrite (MDiv (MMul a b) c) (MMul a (MDiv b c)))
;(rewrite (MMul a (MDiv b c)) (MDiv (MMul a b) c))

; Distributive
;(rewrite (MMul a (MAdd b c)) (MAdd (MMul a b) (MMul a c)))
;(rewrite (MDiv (MAdd a b) c) (MAdd (MDiv a c) (MDiv b c)))

; Constant folding
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)))
(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)))
(rewrite (MMul (MNum a) (MNum b)) (MNum (* a b)) :when ((< a 10000) (< b 10000)))
(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))))
(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)))
(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)))
(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)))
;(rewrite (MOr (MNum a) (MNum b)) (MNum (| a b)))

; Factoring
;(rewrite (MAdd (MMul a b) (MMul a c)) (MMul a (MAdd b c)))
;(rewrite (MAdd a a) (MMul (MNum 2) a))

; Simple reductions
(rewrite (MAdd a (MNum 0)) a)
(rewrite (MMul a (MNum 1)) a)
(rewrite (MMul a (MNum 0)) (MNum 0))
(rewrite (MDiv a (MNum 1)) a)
(rewrite (MMul (MDiv ?a ?b) ?b) (MFloorTo ?a ?b))
(rewrite (MAdd (MFloorTo ?a ?b) (MMod ?a ?b)) ?a)


; Other
;(rewrite (MAdd (MDiv a b) c) (MDiv (MAdd a (MMul c b)) b))

; Replacement
(rewrite (MReplace ?x ?y ?z) ?z :when ((= ?x ?y)))
(rewrite (MReplace (MAdd ?a ?b) ?x ?y) (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MSub ?a ?b) ?x ?y) (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMul ?a ?b) ?x ?y) (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MDiv ?a ?b) ?x ?y) (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMod ?a ?b) ?x ?y) (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMin ?a ?b) ?x ?y) (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMax ?a ?b) ?x ?y) (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MFloorTo ?a ?b) ?x ?y) (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
;; leave numbers unchanged
(rewrite (MReplace (MNum ?n) ?x ?y) (MNum ?n))
(rewrite (MReplace (MAccum ?acc) ?x ?y) (MAccum ?acc))

;; leave other vars unchanged
(rewrite (MReplace (MVar ?v) (MVar ?x) ?y) (MVar ?v) :when ((!= ?v ?x)))


(datatype LoopType (Loop String Math))
(datatype*
 	(Expr
  		; General kernel stuff
     	(GMEM)
     	(LoopIn Expr LoopType Math)
     	(LoopOut Expr LoopType Math)
      	(SMEM)
       	(SMEMLoad Expr Expr)
        (SMEMRead Expr Expr)
        (NewAcc i64) ; define accumulator for a loop

        ; Unary Ops
     	(Exp Expr)
     	(Sin Expr)
      	(Recip Expr)
       	(Neg Expr)

        ; Binary Ops
     	(Add Expr Expr)
     	(Mul Expr Expr)
      	(Max Expr Expr)

        ; search helpers
        (Unary String Expr)
     	(Binary String Expr Expr)
      	(SwapLoops Expr String String) ; Swap two loops, identified by their string
     )
)

; Convert to and from generic unary ops
(rewrite (Exp ?x) (Unary "Exp" ?x))
(rewrite (Unary "Exp" ?x) (Exp ?x))
(rewrite (Sin ?x) (Unary "Sin" ?x))
(rewrite (Unary "Sin" ?x) (Sin ?x))
(rewrite (Recip ?x) (Unary "Recip" ?x))
(rewrite (Unary "Recip" ?x) (Recip ?x))
(rewrite (Neg ?x) (Unary "Neg" ?x))
(rewrite (Unary "Neg" ?x) (Neg ?x))
(rewrite (Add ?a ?b) (Binary "Add" ?a ?b))
(rewrite (Binary "Add" ?a ?b) (Add ?a ?b))
(rewrite (Mul ?a ?b) (Binary "Mul" ?a ?b))
(rewrite (Binary "Mul" ?a ?b) (Mul ?a ?b))
(rewrite (Max ?a ?b) (Binary "Max" ?a ?b))
(rewrite (Binary "Max" ?a ?b) (Max ?a ?b))

; Communative binary ops
(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a))
; distributive/associative skeletons so sums and products re-associate
(rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)))
(rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)))

; remove 1-level loop
(rewrite
 	(LoopOut (Unary ?un (LoopIn ?x (Loop ?loop (MNum 1)) (MVar "z"))) (Loop ?loop (MNum 1)) (MVar "z"))
	(Unary ?un ?x)
)
(rewrite
 	(LoopOut (Binary ?bin (LoopIn ?a (Loop ?loop (MNum 1)) (MVar "z")) (LoopIn ?b (Loop ?loop (MNum 1)) (MVar "z"))) (Loop ?loop (MNum 1)) (MVar "z"))
	(Binary ?bin ?a ?b)
)

; Loop Fusion
;(rewrite (LoopIn (LoopOut ?x ?loop ?st) ?loop ?st) ?x) ; this is causing infinite loops in the e-graph!

; Loop Fission
(rewrite
	(LoopOut (Unary ?second (Unary ?first (LoopIn ?x ?loop ?inSt))) ?loop ?outSt)
	(LoopOut (Unary ?second (LoopIn (LoopOut (Unary ?first (LoopIn ?x ?loop ?inSt)) ?loop  (MVar "z")) ?loop (MVar "z"))) ?loop ?outSt)
)
(rewrite
	(Unary ?second (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?b ?loop ?inBSt)))
	(Unary ?second (LoopIn (LoopOut (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?b ?loop ?inBSt)) ?loop  (MVar "z")) ?loop (MVar "z")))
)
(rewrite
	(Binary ?second (Unary ?first (LoopIn ?a ?loop ?inASt)) (LoopIn ?b ?loop ?inBSt))
	(Binary ?second (LoopIn (LoopOut (Unary ?first (LoopIn ?a ?loop ?inASt)) ?loop  (MVar "z")) ?loop (MVar "z")) (LoopIn ?b ?loop ?inBSt))
)
(rewrite
	(Binary ?second (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?c ?loop ?inCSt)) (LoopIn ?b ?loop ?inBSt))
	(Binary ?second (LoopIn (LoopOut (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?c ?loop ?inCSt)) ?loop  (MVar "z")) ?loop (MVar "z")) (LoopIn ?b ?loop ?inBSt))
)
(rewrite
	(LoopOut (Binary ?second (Unary ?first ?a) ?b) ?loop ?outSt)
	(LoopOut (Binary ?second (LoopIn (LoopOut (Unary ?first ?a) ?loop  (MVar "z")) ?loop (MVar "z")) (LoopIn (LoopOut ?b ?loop  (MVar "z")) ?loop (MVar "z"))) ?loop ?outSt)
)
(rewrite (LoopOut (LoopOut
	(Unary
		?unHere
		(LoopIn (LoopOut ?inpA ?first ?firstStA) ?n ?inASt)
	) ?n ?st) ?lower ?lowerSt)
	(LoopOut (LoopOut
		(Unary
			?unHere
	       	(LoopIn	(LoopIn
	           	(LoopOut (LoopOut ?inpA ?first ?firstStA) ?lower ?lowerSt)
	       	?lower ?lowerSt) ?n ?inASt)
		)
	?n ?st) ?lower ?lowerSt)
)
(rewrite (LoopOut (LoopOut
    (Binary ?binHereA
    	(LoopIn (LoopOut ?inpB ?first ?firstStB) ?n ?inBSt)
        ?a
    ) ?n  ?st) ?lower ?lowerSt)
  	(LoopOut (LoopOut
        (Binary ?binHereA
        	(LoopIn (LoopIn
                (LoopOut (LoopOut ?inpB ?first ?firstStB) ?lower ?lowerSt)
            ?lower ?lowerSt) ?n ?inBSt)
            ?a
        )
    ?n  ?st) ?lower ?lowerSt)
)

; Loop merging
(rewrite
 	(LoopOut (LoopOut
       	(Unary ?merge
      		(LoopIn (LoopIn ?here (Loop ?outerL ?outer) ?outerStride) (Loop ?innerL ?inner) ?innerStride)
        )
    (Loop ?innerL ?inner) ?innerStride) (Loop ?outerL ?outer) ?outerStride)
 	(LoopOut
     	(Unary ?merge
           	(LoopIn ?here
               	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
               	(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
            )
    	)
     	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
		(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
    )
)
(rewrite
 	(LoopOut (LoopOut
       	(Binary
           	?binmerge
           	(LoopIn (LoopIn ?a (Loop ?outerL ?outer) ?outerStrideA) (Loop ?innerL ?inner) ?innerStrideA)
           	(LoopIn (LoopIn ?b (Loop ?outerL ?outer) ?outerStrideB) (Loop ?innerL ?inner) ?innerStrideB)
        )
    (Loop ?innerL ?inner) ?innerStride) (Loop  ?outerL ?outer) ?outerStride)
 	(LoopOut
     	(Binary ?binmerge
        	(LoopIn
             	?a
               	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
               	(MAdd (MReplace ?outerStrideA (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStrideA (MVar "z") (MMod (MVar "z") ?inner)))
            )
            (LoopIn
             	?b
               	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
               	(MAdd (MReplace ?outerStrideB (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStrideB (MVar "z") (MMod (MVar "z") ?inner)))
            )
    	)
     	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
		(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
    )
)

;(rewrite (Unary ?s ?x) (LoopOut (Unary ?s (LoopIn ?x (Loop "_" (MNum 1)) (MVar "z"))) (Loop "_" (MNum 1)) (MVar "z"))) ; add one-level loop

; ───────────────── TESTS ─────────────────
; Common variables
(let tensorA (GMEM))
(let tensorB (GMEM))
(let strideOne (MVar "z"))

; ───────────────── Fission test (1 loop -> 3 sequential loops) ─────────────────
(push)
(let loop (Loop "l" (MNum 1024)))
(let singleLoop (Loop "one" (MNum 1)))
(let inpA
	(LoopIn
		(LoopIn
			(LoopIn
				(LoopIn
					tensorA
				singleLoop strideOne)
			singleLoop strideOne)
		singleLoop strideOne)
	loop strideOne)
)
(let inpB
	(LoopIn
		(LoopIn
			(LoopIn
				(LoopIn
					tensorB
				singleLoop strideOne)
			singleLoop strideOne)
		singleLoop strideOne)
	loop strideOne)
)
(let full
	(LoopOut
		(LoopOut
			(LoopOut
				(LoopOut
					(Add (Sin (Exp inpA)) inpB)
				loop strideOne)
			singleLoop strideOne)
		singleLoop strideOne)
	singleLoop strideOne)
)

(run 5)
;(check (= full (LoopOut (Unary "Sin" (LoopIn (LoopOut (Unary "Exp" (LoopIn tensorA loop strideOne)) loop strideOne) loop strideOne)) loop strideOne)))