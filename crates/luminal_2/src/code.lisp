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


(rewrite  ; 0-1
	(LoopOut (LoopOut ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
	(LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt)
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

;(rewrite (Unary ?s ?x) (LoopOut (Unary ?s (LoopIn ?x (Loop "_" (MNum 1)) (MVar "z"))) (Loop "_" (MNum 1)) (MVar "z"))) ; add one-level loop

; ───────────────── TESTS ─────────────────
; Common variables
(let tensorA (GMEM))
(let tensorB (GMEM))
(let strideOne (MVar "z"))

; ───────────────── Fission test (1 loop -> 3 sequential loops) ─────────────────
(let mLoop (Loop "m" (MNum 10)))
(let nLoop (Loop "n" (MNum 5)))
(let mStride (MMul (MVar "z") (MNum 5)))
(let nStride (MVar "z"))
(let aStrided (LoopIn (LoopIn tensorA mLoop mStride) nLoop nStride))
(let bStrided (LoopIn (LoopIn tensorB mLoop mStride) nLoop nStride))
(let body (Exp (Add (Sin aStrided) bStrided)))
(let full (LoopOut (LoopOut body nLoop nStride) mLoop mStride))
(run 5)