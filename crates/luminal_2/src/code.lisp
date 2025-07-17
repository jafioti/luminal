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
     	(Exp2 Expr)
      	(Log2 Expr)
    	(Sqrt Expr)
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
        (UnpadLoop Expr String) ; Remove a padding loop, identified by it's string
     )
)

; Convert to and from generic unary ops
(rewrite (Exp2 ?x) (Unary "Exp2" ?x))
(rewrite (Unary "Exp2" ?x) (Exp2 ?x))
(rewrite (Log2 ?x) (Unary "Log2" ?x))
(rewrite (Unary "Log2" ?x) (Log2 ?x))
(rewrite (Sqrt ?x) (Unary "Sqrt" ?x))
(rewrite (Unary "Sqrt" ?x) (Sqrt ?x))
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

; remove pad loop
(rewrite
 	(LoopOut (Unary ?un (LoopIn ?x (Loop ?loop (MNum 1)) (MNum 0))) (Loop ?loop (MNum 1)) (MNum 0))
	(Unary ?un ?x)
)
(rewrite
 	(LoopOut (Binary ?bin (LoopIn ?a (Loop ?loop (MNum 1)) (MNum 0)) (LoopIn ?b (Loop ?loop (MNum 1)) (MNum  0))) (Loop ?loop (MNum 1)) (MNum 0))
	(Binary ?bin ?a ?b)
)
; add pad loop
(rewrite
	(LoopOut (Unary ?un ?x) (Loop ?l ?r) ?s)
	(LoopOut (LoopOut (Unary ?un (LoopIn ?x (Loop "newpad" (MNum 1)) (MNum 0))) (Loop "newpad" (MNum 1)) (MNum 0)) (Loop ?l ?r) ?s)
	:when ((!= ?r (MNum 1)) (!= ?s (MNum 0)))
)
(rewrite
	(LoopOut (Binary ?bin ?a ?b) (Loop ?l ?r) ?s)
	(LoopOut (LoopOut (Binary ?bin (LoopIn ?a (Loop "newpad" (MNum 1)) (MNum 0)) (LoopIn ?b (Loop "newpad" (MNum 1)) (MNum 0))) (Loop "newpad" (MNum 1)) (MNum 0)) (Loop ?l ?r) ?s)
	:when ((!= ?r (MNum 1)) (!= ?s (MNum 0)))
)
(rewrite
	(LoopOut (Binary ?bin2 (Binary ?bin ?a ?b) ?c) (Loop ?l ?r) ?s)
	(LoopOut (LoopOut (Binary ?bin2 (Binary ?bin (LoopIn ?a (Loop "newpad" (MNum 1)) (MNum 0)) (LoopIn ?b (Loop "newpad" (MNum 1)) (MNum 0))) (LoopIn ?c (Loop "newpad" (MNum 1)) (MNum 0))) (Loop "newpad" (MNum 1)) (MNum 0)) (Loop ?l ?r) ?s)
	:when ((!= ?r (MNum 1)) (!= ?s (MNum 0)))
)

; Loop Fusion
(rewrite (LoopIn (LoopOut ?x (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st) ?x)

; Loop Fission


; Specialized swap loops
(rewrite
	(LoopOut (LoopOut (Binary ?bin (LoopIn (LoopIn ?a ?outA ?outASt) ?inA ?inASt) (LoopIn (LoopIn ?b ?outB ?outBSt) ?inB ?inBSt)) ?in ?inSt) ?out ?outSt)
	(LoopOut (LoopOut (Binary ?bin (LoopIn (LoopIn ?a ?inA ?inASt) ?outA ?outASt) (LoopIn (LoopIn ?b ?inB ?inBSt) ?outB ?outBSt)) ?out ?outSt) ?in ?inSt)
)
(rewrite
	(LoopOut (LoopOut (Binary ?bin2 (Binary ?bin (LoopIn (LoopIn ?a ?outA ?outASt) ?inA ?inASt) (LoopIn (LoopIn ?b ?outB ?outBSt) ?inB ?inBSt)) (LoopIn (LoopIn ?c ?outC ?outCSt) ?inC ?inCSt)) ?in ?inSt) ?out ?outSt)
	(LoopOut (LoopOut (Binary ?bin2 (Binary ?bin (LoopIn (LoopIn ?a ?inA ?inASt) ?outA ?outASt) (LoopIn (LoopIn ?b ?inB ?inBSt) ?outB ?outBSt)) (LoopIn (LoopIn ?c ?inC ?inCSt) ?outC ?outCSt)) ?out ?outSt) ?in ?inSt)
)

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


;(rewrite (Unary ?s ?x) (LoopOut (Unary ?s (LoopIn ?x (Loop "_" (MNum 1)) (MVar "z"))) (Loop "_" (MNum 1)) (MVar "z"))) ; add one-level loop

{code}
(run {iters})

(let acc_gmem (GMEM "acc_0"))
(let acc_0 (LoopIn acc_gmem (Loop "0" (MNum 1)) (MNum 0)))
(let acc_1 (LoopIn acc_0 (Loop "1" (MNum 5)) (MNum 0)))
(let acc_pad3 (LoopIn acc_1 (Loop "newpad" (MNum 1)) (MNum 0)))
(let acc_pad2 (LoopIn acc_pad3 (Loop "newpad" (MNum 1)) (MNum 0)))
(let acc_2 (LoopIn acc_pad2 (Loop "2" (MNum 4)) (MAccum "a")))

(let weight_gmem (GMEM "Weight Load"))
(let weight_0 (LoopIn weight_gmem (Loop "0" (MNum 1)) (MNum 0)))
(let weight_1 (LoopIn weight_0 (Loop "1" (MNum 5)) (MVar "z")))
(let weight_pad3 (LoopIn weight_1 (Loop "newpad" (MNum 1)) (MNum 0)))
(let weight_pad2 (LoopIn weight_pad3 (Loop "newpad" (MNum 1)) (MNum 0)))
(let weight_2 (LoopIn weight_pad2 (Loop "2" (MNum 4)) (MMul (MVar "z") (MNum 5))))

(let tensor_gmem (GMEM "Tensor Load"))
(let tensor_0 (LoopIn tensor_gmem (Loop "0" (MNum 1)) (MNum 0)))
(let tensor_1 (LoopIn tensor_0 (Loop "1" (MNum 5)) (MNum 0)))
(let tensor_pad3 (LoopIn tensor_1 (Loop "newpad" (MNum 1)) (MNum 0)))
(let tensor_pad2 (LoopIn tensor_pad3 (Loop "newpad" (MNum 1)) (MNum 0)))
(let tensor_2 (LoopIn tensor_pad2 (Loop "2" (MNum 4)) (MVar "z")))

(let add_2 (Add acc_2 (Mul tensor_2 weight_2)))

(let out2_2 (LoopOut add_2 (Loop "2" (MNum 4)) (MAccum "a")))
(let out2_pad2 (LoopOut out2_2 (Loop "newpad" (MNum 1)) (MNum 0)))
(let out2_pad3 (LoopOut out2_pad2 (Loop "newpad" (MNum 1)) (MNum 0)))
(let out2_1 (LoopOut out2_pad3 (Loop "1" (MNum 5)) (MVar "z")))
(let out2_0 (LoopOut out2_1 (Loop "0" (MNum 1)) (MMul (MVar "z") (MNum 5))))

;(check (= out2_0 t28))