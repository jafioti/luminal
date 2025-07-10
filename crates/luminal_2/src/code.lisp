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

; Associative
(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)))
(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)))

; Constant folding
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)))
(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)))
(rewrite (MMul (MNum a) (MNum b)) (MNum (* a b)) :when ((< a 10000) (< b 10000)))
(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))))
(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)))
(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)))
(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)))

; Simple reductions
(rewrite (MAdd a (MNum 0)) a)
(rewrite (MMul a (MNum 1)) a)
(rewrite (MMul a (MNum 0)) (MNum 0))
(rewrite (MDiv a (MNum 1)) a)
(rewrite (MMul (MDiv ?a ?b) ?b) (MFloorTo ?a ?b))
(rewrite (MAdd (MFloorTo ?a ?b) (MMod ?a ?b)) ?a)

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
     	(GMEM String)
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
;(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a))
; distributive/associative skeletons so sums and products re-associate
;(rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)))
;(rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)))

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
(rewrite (LoopIn (LoopOut ?x ?loop ?st) ?loop ?st) ?x
	;:when ((!= ?st (MAccum ?y))) ; don't fuse if we're accumulating that loop
) ; this is causing infinite loops in the e-graph!

; Loop Fission


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

; Split loops
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

(rewrite
 	(LoopOut
  		(Binary ?finbin
    		(LoopIn ?c (Loop ?loopL (MNum ?loop)) ?strideC)
      		(Binary ?spbin
        		(LoopIn ?a (Loop ?loopL (MNum ?loop)) ?strideA)
          		(LoopIn ?b (Loop ?loopL (MNum ?loop)) ?strideB)
            )
        ) (Loop ?loopL (MNum ?loop)) ?stride
    )
 	(LoopOut
     	(LoopOut
      		(Binary ?finbin
        		(LoopIn
                 	(LoopIn
                     	?c
                     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
                     	(MReplace ?strideC (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum tileFactor))
                 	?strideC
                )
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
            )
         	(Loop (+ ?loopL "Split") (MNum tileFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop tileFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum tileFactor)))
    )
 	:when ((> ?loop tileFactor) (= (% ?loop tileFactor) 0))
)

; Loop swapping
(rewrite  ; 0-1
	(LoopOut (LoopOut ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
	(LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt)
	:when ((!= ?innerLoop ?outerLoop))
)
(rewrite ; 0-1
	(SwapLoops (LoopIn (LoopIn ?body (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
)
; propogate
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
(rewrite
	(SwapLoops (NewAcc ?x) ?innerLoop ?outerLoop)
	(NewAcc ?x)
)

;(rewrite (Unary ?s ?x) (LoopOut (Unary ?s (LoopIn ?x (Loop "_" (MNum 1)) (MVar "z"))) (Loop "_" (MNum 1)) (MVar "z"))) ; add one-level loop

{code}
(run 10)