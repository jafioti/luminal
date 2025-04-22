# So you want to run Llama-3 8B via Luminal?
- Awesome! Below are the pre-requisites and step-by-step instructions to get this working on your local machine!

## pre-requisites
- you will need Rust installed on your computer. 
    - If you do not have this yet, [go to the rustup website](https://rustup.rs/) to download this

## step-by-step instructions:
1. Download Llama-3 8B weights from Hugging Face
    -  ```bash
        bash setup.sh
        ```
    - The weights and tokenizer should be saved into the setup directory. This should be the default case, but if they aren't there, make sure to move them here
    - NOTE: This will likely take some time depending on the speed of your internet connection
2. Execute a Sample Llama Prompt
    - Our command is slightly different based off what hardware you're running off
        - if you are on a macbook:
            -  ```bash
                cargo run --release --features metal
                ```
        - if you have an NVIDIA GPU
            -  ```bash
                cargo run --release --features cuda
                ```
        - if you have neither (you just want this to run on CPU)
            -  ```bash
                cargo run --release
                ```
    - This line executes the code found within `src/main.rs` of the Llama example folder 
        - by default we are asking it to generate 256 tokens in response to the merge sort prompt found at `prompts/merge_sort.txt`
    - You should expect the merge sort prompt to be generated in roughly X seconds with a rate of roughly Y tokens per second
        - {INCOMPELTE}
3. Pass through a custom prompt
    - To pass through a custom prompt, we are going to append onto our hardware prompt this flag: `-- --prompt <your_prompt>`
        - the first `--` separates our program arguments from cargo's arguments
    - for example (on Mac):
        -  ```bash
            cargo run --release --features metal -- --prompt '<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
            You are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|> \
            Tell me a story about an AI and a human who work together to save the world<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
            ```
        - {INCOMPELTE}

# Getting An Understanding of the Code
- we start off by parsing our arguments 
    - `let cli_args = CLIArgs::parse();`
- and then loading the tokenizer into memory
    - `let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();`
    - note this comes from an external dependency on the `tokenizer` package
- Now we start to hit the interesting parts, we need to setup our model as a graph within Luminal. Note, the entire section here is not actually inferencing anything but rather telling luminal what to expect at each stage so we can optimize our run
    - we begin by initializing our graph as cx and create an input tensor at the very beginning
    ```rs
        let mut cx = Graph::new();
        let mut input = cx.named_tensor("Input", (1, 's'));
    ```
    - next, we setup placeholder tensors for the key value caches for each layer in our transformer
    ```rs
        let mut cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
            .map(|_| {
                (
                    cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                    cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                )
            })
            .collect();
        cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));
    ```
    - With those setup, we tell it to expect the Llama model shape along with the dimensions of our model weights
    ```rs
        let model = model::Llama::new(&mut cx);
        let mut model_weights = params(&model);
        cx.keep_tensors(&model_weights);
    ```
    - We then say that when we do a forward pass on the model, we expect it to output the logits and a cache_dest variable
    ```rs
        let (logits, mut cache_dest) = model.forward((input, &cache_src));
        let mut logits = logits
            .slice((.., Expression::from('s') - 1.., ..))
            .retrieve();
        cache_dest.keep();
        println!("\t\t - {}ms", now.elapsed().as_millis());
    ```
- Next, we compile the graph based off the hardware we have:
    - the below will return the quantized model weights reference. If we have additional non-CPU resources, we will use the reference to send these weights into memory
    ```rs
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let q_weights = loader::q8_load("setup/llama3-8b.gguf", &model, &mut cx);
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    loader::q8_load("setup/llama3-8b.gguf", &model, &mut cx);
    ```
    - the below then compiles our graph based off the luminal library. Note that for Metal and CUDA we have different precision floats we are using. This can be adjusted based off the quality of your hardware
    ```rs
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            (
                luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(q_weights),
                luminal_metal::BufferCompilers::default(),
            ),
            #[cfg(feature = "cuda")]
            (
                luminal_cuda::CudaCompiler::<f16>::default(),
                luminal_cuda::CudaQuantizedCompiler::<f16>::new(q_weights),
            ),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        (
            &mut input,
            &mut logits,
            &mut cache_src,
            &mut cache_dest,
            &mut model_weights,
        ),
    );
    let cache_src = downstream(&cache_src, &cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());
    ```
    - test yourself: why are we using different precision floats for CUDA and Metal when our weights are quantized?
        - answer: the weights are quantized but not all the intermediary values need to be. We still will use floats to store values as this tends to correlate with better performance
- With our graph compiled, we have one more step before we can inference -- loading the model into memory
    - we will run `cx.execute()` to run our graph the first time
    - note that we don't pass in any real inputs here, just a dummy vector of size (1,1). This step however will load all of the correct model weights into memory and prepare us to inference
    ```rs
    print!("Loading model");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    input.set_dyn(vec![1.], (1, 1));
    cx.set_dyn_dim('t', 1);
    cx.execute();
    logits.drop();
    transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());
    ```
- now we're ready to inference!
    - we begin by tokenizing our inputs and creating the input vector as a vector of f32 values
    ```rs
    let input_ids = tokenizer
        .encode(&cli_args.prompt as &str, false)
        .unwrap()
        .get_ids()
        .to_vec();
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (1, input_ids.len()),
    );
    cx.set_dyn_dim('t', input_ids.len());
    ```
    - just like the before step, we execute the model to load all of the necessary values into the KV cache (this would be the actual pre-filling step). Our output_ids are then chosen based off the largest one to get the suggested token, and we remove the rest from memory. For production LLM use-cases, we would typically use a softmax here, but we did argmax for simplicity
    ```rs
    io::stdout().flush().unwrap();
    let now = Instant::now();
    cx.execute();
    let elapsed_ms = now.elapsed().as_millis();
    println!(
        "\t - {elapsed_ms}ms ({:.2} tok/s, {} prompt tokens)",
        1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64),
        input_ids.len()
    );
    let mut output_ids = vec![argmax(&logits.data())];
    logits.drop();
    ```
    - here we decode the token and print the result to the screen
    ```rs
    print!("{}", cli_args.prompt.white().bold());
    let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
    print!("{initial}",);
    io::stdout().flush().unwrap();
    ```
    - importantly, we end the run by maintaining our cache within the same graph. As we have run attention on new tokens, we want to update our key value cache to account for this new information. Note we are copying data between the tensors here, not changing the pointers
    ```rs
        transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    ```
    - we effectively repeat these steps within the for-loop, inferencing until we reach a stop token or the max-length
    ```rs
    let start_decode = std::time::Instant::now();
    let mut prev_output_len = initial.len();
    for _ in 0..cli_args.gen_tokens {
        input.set_dyn(vec![*output_ids.last().unwrap() as f32], (1, 1));
        cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);
        cx.execute();

        // Sample tokens
        let output_id = argmax(&logits.data());
        logits.drop();
        output_ids.push(output_id);

        // Get the current decoded output
        let current_output = tokenizer.decode(&output_ids, false).unwrap();

        // Print the new substring added to the decoded output
        print!("{}", current_output[prev_output_len..].bright_green());
        io::stdout().flush().unwrap();

        // Update the previous output
        prev_output_len = current_output.len();

        // Swap caches
        transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    }
    ```