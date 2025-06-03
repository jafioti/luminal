use axum::{Router, response::Html, routing::get};
use std::fs;

async fn index() -> Html<String> {
    // Read the file at runtime so you can edit without rebuilding
    let html = fs::read_to_string("static/index.html").expect("`static/index.html` not found");
    Html(html)
}

#[tokio::main]
async fn main() {
    // single route for “/”
    let app = Router::new().route("/", get(index));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
