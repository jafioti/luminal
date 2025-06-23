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
     	(GMEM String)
     	(LoopIn Expr LoopType Math)
     	(LoopOut Expr LoopType Math)
     	(Exp Expr)
     	(Sin Expr)
      	(Recip Expr)
       	(Neg Expr)
     	(Unary String Expr)
      	(ZeroStrideLoad Expr LoopType Math)
     	(Add Expr Expr)
     	(Mul Expr Expr)
      	(Max Expr Expr)
     	(Binary String Expr Expr)
     	(NewAcc i64) ; define accumulator for a loop
     	(AccOut Expr LoopType) ; retrieve the final result accumulator after the loop is complete
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
(rewrite (LoopIn (LoopOut ?x ?loop ?st) ?loop ?st) ?x)

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
; Tile loops
(let tileFactor 8)
(rewrite
 	(LoopOut (Unary ?spun (LoopIn ?x (Loop ?loopL (MNum ?loop)) ?stride)) (Loop ?loopL (MNum ?loop)) ?stride)
 	(LoopOut
     	(LoopOut
         	(Unary ?spun
            	(LoopIn
                 	(LoopIn
                     	?x
                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
                     	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
                     )
                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
                 	?stride
                )
            )
         	(Loop (+ ?loopL "Split") (MNum tileFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
    )
 	:when ((> ?loop tileFactor) (= (% ?loop tileFactor) 0))
)
(rewrite
 	(LoopOut (Binary ?spbin (LoopIn ?a (Loop ?loopL (MNum ?loop)) ?strideA) (LoopIn ?b (Loop ?loopL (MNum ?loop)) ?strideB)) (Loop ?loopL (MNum ?loop)) ?stride)
 	(LoopOut
     	(LoopOut
         	(Binary ?spbin
            	(LoopIn
                 	(LoopIn
                     	?a
                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
                     	(MReplace ?strideA (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
                 	?strideA
                )
                (LoopIn
                 	(LoopIn
                     	?b
                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
                     	(MReplace ?strideB (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
                 	?strideB
                )
            )
         	(Loop (+ ?loopL "Split") (MNum tileFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
    )
 	:when ((> ?loop tileFactor) (= (% ?loop tileFactor) 0))
)
(rewrite ; SHORTCUT: immediately split loops with body of Binary(Binary(a, b), c)
 	(LoopOut
  		(Binary ?bin2
    		(Binary ?bin1
	    		(LoopIn ?a (Loop ?loopL (MNum ?loop)) ?strideA)
	      		(LoopIn ?b (Loop ?loopL (MNum ?loop)) ?strideB)
	        )
	        (LoopIn ?c (Loop ?loopL (MNum ?loop)) ?strideC)
        )
    (Loop ?loopL (MNum ?loop)) ?stride)
 	(LoopOut
     	(LoopOut
         	(Binary ?bin2
          		(Binary ?bin1
	            	(LoopIn
	                 	(LoopIn
	                     	?a
	                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
	                     	(MReplace ?strideA (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
	                    )
	                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
	                 	?strideA
	                )
	                (LoopIn
	                 	(LoopIn
	                     	?b
	                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
	                     	(MReplace ?strideB (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
	                    )
	                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
	                 	?strideB
	                )
	            )
				(LoopIn
                 	(LoopIn
                     	?c
                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
                     	(MReplace ?strideC (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
                 	?strideC
                )
			)
         	(Loop (+ ?loopL "Split") (MNum tileFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
    )
 	:when ((> ?loop 512) (= (% ?loop tileFactor) 0))
)

; Swap dimensions
(rewrite  ; 0-1
	(LoopOut (LoopOut ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
	(LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt)
)
(rewrite  ; 0-2 SHORTCUT
	(LoopOut (LoopOut (LoopOut ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?midLoop ?midSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
	(LoopOut (LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?outerLoop ?outerLoopAmt) ?outerSt) ?midLoop ?midSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt)
)
(rewrite ; 0-1
	(SwapLoops (LoopIn (LoopIn ?body (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
)
(rewrite ; 0-2 SHORTCUT
	(SwapLoops (LoopIn (LoopIn (LoopIn ?body (Loop ?outerLoop ?outerLoopAmt) ?outerSt) ?midLoop ?midSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn (LoopIn ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?midLoop ?midSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
)
(rewrite
	(SwapLoops (LoopIn ?body (Loop ?otherLoop ?otherLoopAmt) ?otherSt) ?innerLoop ?outerLoop)
	(LoopIn (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?otherLoop ?otherLoopAmt) ?otherSt)
	:when ((!= ?innerLoop ?otherLoop))
)
(rewrite
	(SwapLoops (LoopOut ?body (Loop ?otherLoop ?otherLoopAmt) ?otherSt) ?innerLoop ?outerLoop)
	(LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?otherLoop ?otherLoopAmt) ?otherSt)
)
(rewrite
	(SwapLoops (Unary ?un ?body) ?innerLoop ?outerLoop)
	(Unary ?un (SwapLoops ?body ?innerLoop ?outerLoop))
)
(rewrite
	(SwapLoops (Binary ?bin ?bodyA ?bodyB) ?innerLoop ?outerLoop)
	(Binary ?bin (SwapLoops ?bodyA ?innerLoop ?outerLoop) (SwapLoops ?bodyB ?innerLoop ?outerLoop))
)

; Exp-Sum-Max Online Trick
(rewrite
	(LoopOut (Add (Exp (Add
		?online
		(Neg (LoopIn (LoopOut (Max (LoopIn ?maxAcc ?loop (MAccum ?maxAccName)) ?online) ?loop (MAccum ?maxAccName)) ?loop (MNum 0)))
	)) (LoopIn ?sumAcc ?loop (MAccum ?sumAccName))) ?loop (MAccum ?sumAccName))
	(LoopOut
		(Add
			(Mul
				(LoopIn ?sumAcc ?loop (MAccum ?sumAccName))
				(Exp (Add (LoopIn ?maxAcc ?loop (MAccum ?maxAccName)) (Neg (Max (LoopIn ?maxAcc ?loop (MAccum ?maxAccName)) ?online))))
			)
			(Exp (Add ?online (Neg (Max (LoopIn ?maxAcc ?loop (MAccum ?maxAccName)) ?online))))
		)
	?loop (MAccum ?maxAccName))
)

; Online softmax -> weighted sum trick
(rewrite
	(LoopOut
		(LoopOut
			(Add
				(Mul
					(LoopIn
						(Mul
							(Exp
								(Add
									(LoopIn (LoopOut ?dot ?kLoop (MVar "z")) ?kLoop (MVar "z"))
									(Neg (LoopIn (LoopOut ?max_acc ?kLoop (MAccum ?maxAcc)) ?kLoop (MNum 0)))
								)
							)
							(Recip (LoopIn (LoopOut (Add (Mul ?exp_sum_acc (Exp (Add ?redot (Neg ?renew_max)))) ?weight) ?kLoop (MAccum ?preAcc)) ?kLoop (MNum 0)))
						)
						?vLoop (MNum 0)
					)
					(LoopIn (LoopIn ?val ?kLoop ?valRow) ?vLoop (MVar "z"))
				)
				(LoopIn (LoopIn ?output_acc ?kLoop (MAccum ?acc)) ?vLoop (MVar "z"))
			)
			?vLoop (MVar "z")
		)
		?kLoop (MAccum ?acc)
	)
	(LoopOut
		(Mul
			(LoopIn
				(LoopOut
					(LoopOut
						(Add
							(Mul
								(LoopIn (LoopIn ?output_acc ?kLoop (MAccum ?acc)) ?vLoop (MVar "z"))
								(LoopIn (Exp (Add ?redot (Neg ?renew_max))) ?vLoop (MNum 0))
							)
							(Mul
								(LoopIn ?weight ?vLoop (MNum 0))
								(LoopIn (LoopIn ?val ?kLoop ?valRow) ?vLoop (MVar "z"))
							)
						)
						?vLoop (MVar "z")
					)
					?kLoop (MAccum ?acc)
				)
				?vLoop (MVar "z")
			)
			(Recip (LoopIn (LoopOut (Add (Mul ?exp_sum_acc (Exp (Add ?redot (Neg ?renew_max)))) ?weight) ?kLoop (MAccum ?preAcc)) ?vLoop (MNum 0)))
		)
		?vLoop (MVar "z")
	)
)

; add zero stride load
(rewrite
	(LoopIn (LoopIn (LoopIn ?x (Loop ?outer ?n) (MNum 0)) ?mid ?midSt) (Loop ?inner ?n) ?innerSt)
	(LoopIn (ZeroStrideLoad (LoopIn (LoopIn ?x (Loop ?outer ?n) (MNum 0)) ?mid ?midSt) (Loop ?outer ?n) ?innerSt) (Loop ?inner ?n) ?innerSt)
	:when ((!= ?innerSt (MNum 0)))
)


; ───────────────── TESTS ─────────────────
; Common variables
(let tensorA (GMEM "A"))
(let tensorB (GMEM "B"))
(let tensorC (GMEM "C"))
(let strideOne (MVar "z"))

; ───────────────── Fission test (1 loop -> 3 sequential loops) ─────────────────
(push)
(let loop (Loop "l" (MNum 4096)))
(let full (LoopOut (Exp (Sin (Add (LoopIn tensorA loop strideOne) (Sin (LoopIn tensorB loop strideOne))))) loop strideOne))
(run 10)
(let part0 (LoopOut (Sin (LoopIn tensorB loop strideOne)) loop strideOne))
(let part1 (LoopOut (Add (LoopIn tensorA loop strideOne) (LoopIn part0 loop strideOne)) loop strideOne))
(let part2 (LoopOut (Sin (LoopIn part1 loop strideOne)) loop strideOne))
(let part3 (LoopOut (Exp (LoopIn part2 loop strideOne)) loop strideOne))
(check (= full part3))