use crate::model::Mistral;

mod model;

fn main() {
    let mistral = Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model").unwrap();

    let prompt = "Hello Mistral";

    let input = mistral.encode(prompt);
    println!("{:?}", input)
}
