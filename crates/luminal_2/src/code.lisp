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
       	(TileLoop Expr String) ; Tile a loop, identified by it's string
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


; Loop tiling
(rewrite
	(LoopOut ?body (Loop ?loop (MNum ?range)) ?stride)
	(LoopOut
		(LoopOut
			(TileLoop ?body ?loop)
			(Loop (+ ?loop "_tile") (MNum 8))
			?stride
		)
		(Loop ?loop (MNum (/ ?range 8)))
		(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))
	)
	:when ((> ?range 8) (= (% ?range 8) 0))
)
(rewrite
	(TileLoop (LoopIn ?body (Loop ?loop (MNum ?range)) ?stride) ?loop)
	(LoopIn (LoopIn ?body (Loop ?loop (MNum (/ ?range 8))) (MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))) (Loop (+ ?loop "_tile") (MNum 8)) ?stride)
	:when ((> ?range 8) (= (% ?range 8) 0))
)
; propogate
(rewrite
	(TileLoop (LoopIn ?body (Loop ?other ?range) ?stride) ?loop)
	(LoopIn (TileLoop ?body ?loop) (Loop ?other ?range) ?stride)
	:when ((!= ?loop ?other))
)
(rewrite
	(TileLoop (LoopOut ?body (Loop ?other ?range) ?stride) ?loop)
	(LoopOut (TileLoop ?body ?loop) (Loop ?other ?range) ?stride)
)
(rewrite
	(TileLoop (LoopIn (LoopIn ?body (Loop ?otherOther ?rangeOther) ?strideOther) (Loop ?other ?range) ?stride) ?loop)
	(LoopIn (LoopIn (TileLoop ?body ?loop) (Loop ?otherOther ?rangeOther) ?strideOther) (Loop ?other ?range) ?stride)
	:when ((!= ?loop ?other) (!= ?loop ?otherOther))
)
(rewrite
	(TileLoop (LoopOut (LoopOut ?body (Loop ?otherOther ?rangeOther) ?strideOther) (Loop ?other ?range) ?stride) ?loop)
	(LoopOut (LoopOut (TileLoop ?body ?loop)  (Loop ?otherOther ?rangeOther) ?strideOther) (Loop ?other ?range) ?stride)
)
(rewrite
	(TileLoop (Unary ?un ?body) ?loop)
	(Unary ?un (TileLoop ?body ?loop))
)
(rewrite
	(TileLoop (Binary ?bin ?bodyA ?bodyB) ?loop)
	(Binary ?bin (TileLoop ?bodyA ?loop) (TileLoop ?bodyB ?loop))
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
	(SwapLoops (LoopIn (LoopIn ?body (Loop ?otherOtherLoop ?otherOtherLoopAmt) ?otherOtherSt) (Loop ?otherLoop ?otherLoopAmt) ?otherSt) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?otherOtherLoop ?otherOtherLoopAmt) ?otherOtherSt) (Loop ?otherLoop ?otherLoopAmt) ?otherSt)
	:when ((!= ?innerLoop ?otherLoop) (!= ?innerLoop ?otherOtherLoop))
)
(rewrite
	(SwapLoops (LoopOut (LoopOut ?body (Loop ?otherOtherLoop ?otherOtherLoopAmt) ?otherOtherSt) (Loop ?otherLoop ?otherLoopAmt) ?otherSt) ?innerLoop ?outerLoop)
	(LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?otherOtherLoop ?otherOtherLoopAmt) ?otherOtherSt) (Loop ?otherLoop ?otherLoopAmt) ?otherSt)
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

{code}
(run 10)

(let acc_gmem (GMEM "Acc"))
(let acc_pad0 (LoopIn acc_gmem (Loop "pad" (MNum 1)) (MNum 0)))
(let acc_pad1 (LoopIn acc_pad0 (Loop "pad" (MNum 1)) (MNum 0)))
(let acc_m (LoopIn acc_pad1 (Loop "m" (MNum 8)) (MMul (MMul (MVar "z") (MNum 8)) (MNum 64))))
(let acc_n (LoopIn acc_m (Loop "n" (MNum 8)) (MMul (MVar "z") (MNum 8))))
(let acc_m_tiled (LoopIn acc_n (Loop "m_tile" (MNum 8)) (MMul (MVar "z") (MNum 64))))
(let acc_n_tiled (LoopIn acc_m_tiled (Loop "n_tile" (MNum 8)) (MVar "z")))
(let acc_k (LoopIn acc_n_tiled (Loop "k" (MNum 8)) (MAccum "a")))
(let acc_k_tiled (LoopIn acc_k (Loop "k_tile" (MNum 8)) (MAccum "a")))

(let b_gmem (GMEM "B"))
(let b_pad0 (LoopIn b_gmem (Loop "pad" (MNum 1)) (MNum 0)))
(let b_pad1 (LoopIn b_pad0 (Loop "pad" (MNum 1)) (MNum 0)))
(let b_m (LoopIn b_pad1 (Loop "m" (MNum 8)) (MNum 0)))
(let b_n (LoopIn b_m (Loop "n" (MNum 8)) (MMul (MVar "z") (MNum 8))))
(let b_m_tiled (LoopIn b_n (Loop "m_tile" (MNum 8)) (MNum 0)))
(let b_n_tiled (LoopIn b_m_tiled (Loop "n_tile" (MNum 8)) (MVar "z")))
(let b_k (LoopIn b_n_tiled (Loop "k" (MNum 8)) (MMul (MMul (MVar "z") (MNum 8)) (MNum 64))))
(let b_k_tiled (LoopIn b_k (Loop "k_tile" (MNum 8)) (MMul (MVar "z") (MNum 64))))

(let a_gmem (GMEM "A"))
(let a_pad0 (LoopIn a_gmem (Loop "pad" (MNum 1)) (MNum 0)))
(let a_pad1 (LoopIn a_pad0 (Loop "pad" (MNum 1)) (MNum 0)))
(let a_m (LoopIn a_pad1 (Loop "m" (MNum 8)) (MMul (MMul (MVar "z") (MNum 8)) (MNum 64))))
(let a_n (LoopIn a_m (Loop "n" (MNum 8)) (MNum 0)))
(let a_m_tiled (LoopIn a_n (Loop "m_tile" (MNum 8)) (MMul (MVar "z") (MNum 64))))
(let a_n_tiled (LoopIn a_m_tiled (Loop "n_tile" (MNum 8)) (MNum 0)))
(let a_k (LoopIn a_n_tiled (Loop "k" (MNum 8)) (MMul (MVar "z") (MNum 8))))
(let a_k_tiled (LoopIn a_k (Loop "k_tile" (MNum 8)) (MVar "z")))

(let out (Add acc_k_tiled (Mul b_k_tiled a_k_tiled)))

(let out_k_tiled (LoopOut out (Loop "k_tile" (MNum 8)) (MAccum "a")))
(let out_k (LoopOut out_k_tiled (Loop "k" (MNum 8)) (MAccum "a")))
(let out_n_tiled (LoopOut out_k (Loop "n_tile" (MNum 8)) (MVar "z")))
(let out_mtiled (LoopOut out_n_tiled (Loop "m_tile" (MNum 8)) (MMul (MVar "z") (MNum 64))))
(let out_n (LoopOut out_mtiled (Loop "n" (MNum 8)) (MMul (MVar "z") (MNum 8))))
(let out_m (LoopOut out_n (Loop "m" (MNum 8)) (MMul (MMul (MVar "z") (MNum 8)) (MNum 64))))
(let out_pad0 (LoopOut out_m (Loop "pad" (MNum 1)) (MNum 0)))
(let out_pad1 (LoopOut out_pad0 (Loop "pad" (MNum 1)) (MNum 0)))

(check (= out_pad1 t24))