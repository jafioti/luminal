; -------- SYMBOLIC ALGEBRA -------

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

; reduce multi-dim squeezed indexing into simple multiplicative indexing
(rewrite
  (MAdd (MMul (MNum (* d n2)) (MMod (MDiv ?v (MNum d)) (MNum m)))
        (MMul (MNum n2) (MMod ?v (MNum d))))
  (MMul ?v (MNum n2))
)

; -------- IR --------

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

    	; propogation pattern helpers
     	(PropOneArg String IR String) ; Generic prop one arg back
     	(PropTwoArgs String IR String String) ; Generic prop two args back
     )
)

; -------------- HELPERS ---------------

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
; propogation patterns
(rewrite (SwapLoops ?expr ?a ?b) (PropTwoArgs "SwapLoops" ?expr ?a ?b))
(rewrite (PropTwoArgs "SwapLoops" ?expr ?a ?b) (SwapLoops ?expr ?a ?b))
;(rewrite (TileLoop ?expr ?loop) (PropOneArg "TileLoop" ?expr ?loop))
;(rewrite (PropOneArg "TileLoop" ?expr ?loop)  (TileLoop ?expr ?loop))
(rewrite (UnpadLoop ?expr ?loop) (PropOneArg "UnpadLoop" ?expr ?loop))
(rewrite (PropOneArg "UnpadLoop" ?expr ?loop)  (UnpadLoop ?expr ?loop))
(rewrite (MergeLoops ?expr ?loopA ?loopB) (PropTwoArgs "MergeLoops" ?expr ?loopA ?loopB))
(rewrite (PropTwoArgs "MergeLoops" ?expr ?loopA ?loopB)  (MergeLoops ?expr ?loopA ?loopB))

; propogation helpers
(rewrite
	(PropOneArg ?prop (LoopIn ?body (Loop ?other ?range) ?stride) ?arg)
	(LoopIn (PropOneArg ?prop ?body ?arg) (Loop ?other ?range) ?stride)
)
(rewrite
	(PropOneArg ?prop (LoopOut ?body (Loop ?other ?range) ?stride) ?arg)
	(LoopOut (PropOneArg ?prop ?body ?arg) (Loop ?other ?range) ?stride)
)
(rewrite
	(PropOneArg ?prop (Unary ?un ?body) ?arg)
	(Unary ?un (PropOneArg ?prop ?body ?arg))
)
(rewrite
	(PropOneArg ?prop (Binary ?bin ?bodyA ?bodyB) ?arg)
	(Binary ?bin (PropOneArg ?prop ?bodyA ?arg) (PropOneArg ?prop ?bodyB ?arg))
)
(rewrite
	(PropTwoArgs ?prop (LoopIn ?body (Loop ?other ?range) ?stride) ?arg1 ?arg2)
	(LoopIn (PropTwoArgs ?prop ?body ?arg1 ?arg2) (Loop ?other ?range) ?stride)
)
(rewrite
	(PropTwoArgs ?prop (LoopOut ?body (Loop ?other ?range) ?stride) ?arg1 ?arg2)
	(LoopOut (PropTwoArgs ?prop ?body ?arg1 ?arg2) (Loop ?other ?range) ?stride)
)
(rewrite
	(PropTwoArgs ?prop (Unary ?un ?body) ?arg1 ?arg2)
	(Unary ?un (PropTwoArgs ?prop ?body ?arg1 ?arg2))
)
(rewrite
	(PropTwoArgs ?prop (Binary ?bin ?bodyA ?bodyB) ?arg1 ?arg2)
	(Binary ?bin (PropTwoArgs ?prop ?bodyA ?arg1 ?arg2) (PropTwoArgs ?prop ?bodyB ?arg1 ?arg2))
)

; Communative binary ops
;(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a))
; distributive/associative skeletons so sums and products re-associate
;(rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)))
;(rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)))

; ---------- RULES ----------

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

; Loop Fusion
;(rewrite (LoopIn (LoopOut ?x (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st) ?x)

; Specialized swap loops
(rewrite
	(LoopOut (LoopOut (Binary ?bin (LoopIn (LoopIn ?a (Loop ?outAL ?outA) ?outASt) (Loop ?inAL ?inA) ?inASt) (LoopIn (LoopIn ?b (Loop ?outBL ?outB) ?outBSt) (Loop ?inBL ?inB) ?inBSt)) (Loop ?inL ?in) ?inSt) (Loop ?outL ?out) ?outSt)
	(LoopOut (LoopOut (Binary ?bin (LoopIn (LoopIn ?a (Loop (+ ?inAL "sw") ?inA) ?inASt) (Loop (+ ?outAL "sw") ?outA) ?outASt) (LoopIn (LoopIn ?b (Loop (+ ?inBL "sw") ?inB) ?inBSt) (Loop (+ ?outBL "sw") ?outB) ?outBSt)) (Loop (+ ?outL "sw") ?out) ?outSt) (Loop (+ ?inL "sw") ?in) ?inSt)
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
	:when ((> ?range tileFactor) (= (% ?range tileFactor) 0))
)
; propogation
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





; Merging
(rewrite
 	(LoopOut (LoopOut ?ir (Loop ?innerL ?inner) ?innerStride) (Loop ?outerL ?outer) ?outerStride)
 	(LoopOut (MergeLoops ?ir ?innerL ?outerL)
     	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
		(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
    )
)
(rewrite (MergeLoops (LoopIn (LoopIn ?ir (Loop ?outerL ?outer) ?outerStride) (Loop ?innerL ?inner) ?innerStride) ?innerL ?outerL)
	(LoopIn ?ir
		(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
		(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
	)
)

{code}
(run {iters})


(let acc_gmem (GMEM "acc_0"))
(let acc_0 (LoopIn acc_gmem (Loop "0" (MNum 4)) (MNum 0)))
(let acc_pad1 (LoopIn acc_0 (Loop "-pad1-" (MNum 1)) (MNum 0)))
(let acc_pad2 (LoopIn acc_pad1 (Loop "-pad2-" (MNum 1)) (MNum 0)))
(let acc_2 (LoopIn acc_pad2 (Loop "2" (MNum 2)) (MAccum "a")))

(let a_gmem (GMEM "A Load"))

(let a_0
    (LoopIn a_gmem
        (Loop "0" (MNum 4))
        (MAdd
            (MMul (MNum 2) (MMod (MDiv (MMul (MVar "z") (MNum 2)) (MNum 4)) (MNum 2)))
            (MMod (MMul (MVar "z") (MNum 2)) (MNum 2))
        )
    )
)
(let a_pad1 (LoopIn a_0 (Loop "newpad" (MNum 1)) (MNum 0)))
(let a_pad2 (LoopIn a_pad1 (Loop "newpad" (MNum 1)) (MNum 0)))
(let a_1
    (LoopIn a_pad2
        (Loop "0_tile" (MNum 2))
        (MAdd
            (MMul (MNum 2) (MMod (MDiv (MVar "z") (MNum 4)) (MNum 2)))
            (MMod (MVar "z") (MNum 2))
        )
    )
)

(let b_gmem (GMEM "B Load"))

(let b_0
    (LoopIn b_gmem
        (Loop "0" (MNum 4))
        (MAdd
            (MMod (MDiv (MMul (MVar "z") (MNum 2)) (MNum 2)) (MNum 2))
            (MMul (MNum 2) (MMod (MMul (MVar "z") (MNum 2)) (MNum 2)))
        )
    )
)
(let b_pad1 (LoopIn b_0 (Loop "newpad" (MNum 1)) (MNum 0)))
(let b_pad2 (LoopIn b_pad1 (Loop "newpad" (MNum 1)) (MNum 0)))
(let b_1
    (LoopIn b_pad2
        (Loop "0_tile" (MNum 2))
        (MAdd
            (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))
            (MMul (MNum 2) (MMod (MVar "z") (MNum 2)))
        )
    )
)

(let mul_0 (Mul a_1 b_1))
(let add_final (Add acc_2 mul_0))

(let out2 (LoopOut add_final (Loop "2" (MNum 2)) (MAccum "a")))
(let out_pad2 (LoopOut out2 (Loop "-pad2-" (MNum 1)) (MVar "z")))
(let out_pad1 (LoopOut out_pad2 (Loop "-pad1-" (MNum 1)) (MVar "z")))
(let out0 (LoopOut out_pad1 (Loop "0" (MNum 4)) (MVar "z")))

(let at0 (GMEM "B Load"))
(let at1 (LoopIn at0 (Loop "" (MNum 8)) (MAdd (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2)))))))
(let at2 (LoopIn at1 (Loop "" (MNum 1)) (MNum 0)))
(let at3 (GMEM "A Load"))
(let at4 (LoopIn at3 (Loop "" (MNum 8)) (MAdd (MMul (MNum 2) (MMod (MDiv (MVar "z") (MNum 4)) (MNum 2))) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))))
(let at5 (LoopIn at4 (Loop "" (MNum 1)) (MNum 0)))
(let at6 (Mul at2 at5))
(let at7 (LoopOut at6 (Loop "" (MNum 1)) (MNum 0)))
(let at8 (LoopOut at7 (Loop "" (MNum 8)) (MVar "z")))
(let at9 (LoopIn at8 (Loop "" (MNum 4)) (MAdd (MAdd (MMul (MNum 4) (MMod (MDiv (MAdd (MMul (MNum 4) (MMod (MDiv (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 2))) (MMul (MNum 2) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2)))) (MNum 2)) (MNum 2))) (MMul (MNum 2) (MMod (MAdd (MMul (MNum 4) (MMod (MDiv (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 2))) (MMul (MNum 2) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2)))) (MNum 2)))) (MAdd (MMul (MNum 2) (MMul (MNum 2) (MMod (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)))) (MMul (MNum 2) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2)))))))
(let at10 (LoopOut at9 (Loop "" (MNum 2)) (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2)))))))
(let at11 (LoopOut at10 (Loop "" (MNum 2)) (MAdd (MMul (MNum 2) (MMul (MNum 2) (MMod (MDiv (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2)) (MNum 2)))) (MMul (MNum 2) (MMod (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2))))))
(let at12 (LoopIn at11 (Loop "" (MNum 2)) (MAdd (MMul (MNum 2) (MMul (MNum 2) (MMod (MDiv (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2)) (MNum 2)))) (MMul (MNum 2) (MMod (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2))))))
(let at13 (LoopIn at12 (Loop "" (MNum 2)) (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2)))))))
(let at14 (LoopOut at13 (Loop "" (MNum 4)) (MAdd (MAdd (MMul (MNum 4) (MMod (MDiv (MAdd (MMul (MNum 4) (MMod (MDiv (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 2))) (MMul (MNum 2) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2)))) (MNum 2)) (MNum 2))) (MMul (MNum 2) (MMod (MAdd (MMul (MNum 4) (MMod (MDiv (MDiv (MVar "z") (MNum 2)) (MNum 2)) (MNum 2))) (MMul (MNum 2) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2)))) (MNum 2)))) (MAdd (MMul (MNum 2) (MMul (MNum 2) (MMod (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)))) (MMul (MNum 2) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2)))))))
(let at15 (LoopIn at14 (Loop "" (MNum 2)) (MAdd (MMul (MNum 2) (MMul (MNum 2) (MMod (MDiv (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2)) (MNum 2)))) (MMul (MNum 2) (MMod (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2))))))
(let at16 (LoopOut at15 (Loop "" (MNum 2)) (MAdd (MMul (MNum 2) (MMul (MNum 2) (MMod (MDiv (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2)) (MNum 2)))) (MMul (MNum 2) (MMod (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2))))) (MNum 2))))))
(let at17 (LoopIn at16 (Loop "" (MNum 4)) (MAdd (MMul (MNum 4) (MMod (MDiv (MVar "z") (MNum 2)) (MNum 2))) (MMul (MNum 2) (MAdd (MMul (MDiv (MMod (MVar "z") (MNum 2)) (MNum 2)) (MNum 2)) (MMod (MMod (MVar "z") (MNum 2)) (MNum 2)))))))
(let at18 (LoopIn at17 (Loop "" (MNum 1)) (MNum 0)))
(let at19 (LoopIn at18 (Loop "" (MNum 1)) (MNum 0)))
(let at20 (LoopIn at19 (Loop "" (MNum 2)) (MVar "z")))
(let at21 (LoopIn at20 (Loop "" (MNum 1)) (MNum 0)))
(let at22 (GMEM "acc_0"))
(let at23 (LoopIn at22 (Loop "" (MNum 4)) (MNum 0)))
(let at24 (LoopIn at23 (Loop "" (MNum 1)) (MNum 0)))
(let at25 (LoopIn at24 (Loop "" (MNum 1)) (MNum 0)))
(let at26 (LoopIn at25 (Loop "" (MNum 2)) (MAccum "a")))
(let at27 (LoopIn at26 (Loop "" (MNum 1)) (MNum 0)))
(let at28 (Add at21 at27))
(let at29 (LoopOut at28 (Loop "" (MNum 1)) (MNum 0)))
(let at30 (LoopOut at29 (Loop "" (MNum 2)) (MAccum "a")))
(let at31 (LoopOut at30 (Loop "" (MNum 1)) (MVar "z")))
(let at32 (LoopOut at31 (Loop "" (MNum 1)) (MVar "z")))
(let at33 (LoopOut at32 (Loop "" (MNum 4)) (MVar "z")))

(rewrite (Loop ?name ?range) (Loop "" ?range))

;(run 1)

;(check (= at33 t19))