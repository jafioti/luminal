use luminal::prelude::*;

fn main() {
    println!("Testing arange implementation...");
    
    // Test 1: Simple constant
    {
        let mut cx = Graph::new();
        let a = cx.constant(1.0).retrieve();
        cx.execute();
        println!("Constant 1.0: {:?}", a.data());
    }
    
    // Test 2: Expand dimension
    {
        let mut cx = Graph::new();
        let a = cx.constant(1.0).expand_dim(0, 5).retrieve();
        cx.execute();
        println!("Constant 1.0 expanded to 5: {:?}", a.data());
    }
    
    // Test 3: Test cumsum directly
    {
        let mut cx = Graph::new();
        let a = cx.constant(1.0).expand_dim(0, 5);
        let b = a.cumsum_last_dim().retrieve();
        cx.execute();
        println!("Cumsum of [1,1,1,1,1]: {:?}", b.data());
    }
    
    // Test 4: Test arange(5)
    {
        let mut cx = Graph::new();
        let a = cx.arange(5).retrieve();
        cx.execute();
        println!("arange(5): {:?}", a.data());
    }
    
    // Test 5: Step by step arange implementation
    {
        let mut cx = Graph::new();
        let ones = cx.constant(1.0).expand_dim(0, 5);
        let cumsum = ones.cumsum_last_dim();
        let result = (cumsum - 1.0).retrieve();
        cx.execute();
        println!("Manual arange(5): {:?}", result.data());
    }
} 