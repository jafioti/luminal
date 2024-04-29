use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// src/chat.rs
use crate::llama::setup::Model; // Import the Model struct

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
}

#[derive(Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Debug)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "user")]
    User,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub fn apply_chat_template(messages: Vec<Message>) -> String {
    let mut output = "<|begin_of_text|>".to_string();
    for message in messages {
        output += "<|start_header_id|>";
        if message.role == Role::Assistant {
            output += "assistant"
        } else if message.role == Role::User {
            output += "user"
        } else if message.role == Role::System {
            output += "system"
        }
        output += "<|end_header_id|>";
        output += "\n";
        output += message.content.as_str();
        output += "<|eot_id|>";
    }
    output
}

/// Respond to chat request
pub async fn respond_chat_request(model: &mut Model, request: ChatRequest) -> ChatResponse {
    let created = Utc::now().timestamp();
    let raw_uuid = Uuid::new_v4();
    let id = format!("chatcmpl-{}", raw_uuid);

    let mut prompt = apply_chat_template(request.messages);
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n";
    let prompt_tokens = model.tokenizer.encode(prompt.clone(), false).unwrap();
    let prompt_tokens = prompt_tokens.get_ids();
    let prompt_tokens = prompt_tokens.len();
    println!("Prompt: {:?}", prompt);

    // Generate
    let mut completion = vec![];
    model.generate(&prompt, |token| {
        completion.push(token);
        println!(" {}", token);

        // TODO: Remove this hack, it's a sign the generation is not working
        token != 97720
    });
    let completion_str = model.tokenizer.decode(&completion, false).unwrap();
    let completion_tokens = completion.len();

    let response = ChatResponse {
        id,
        created,
        object: "chat.completion".to_string(),
        model: "meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: completion_str,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens,
            completion_tokens,
        },
    };

    response
}
