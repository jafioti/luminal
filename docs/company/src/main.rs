use axum::{
    Router,
    response::{Html, Redirect},
    routing::get,
};

async fn index() -> Html<String> {
    Html(include_str!("../static/index.html").to_string())
}

async fn intro_quiz() -> Html<String> {
    Html(include_str!("../static/intro_quiz.html").to_string())
}

async fn redirect() -> Redirect {
    Redirect::to("https://luminalai.com")
}

#[tokio::main]
async fn main() {
    // single route for “/”
    let app = Router::new()
        .route("/", get(index))
        .route("/intro_quiz", get(intro_quiz))
        .route("/docs/introduction", get(redirect));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
