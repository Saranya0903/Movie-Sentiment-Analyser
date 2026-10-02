import gradio as gr
import requests
import numpy as np
import matplotlib.pyplot as plt


API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# HELPERS
# ============================================================

def make_attention_html(tokens, weights):

    if not tokens or not weights:
        return """
        <div class="empty-state">
            🔎 Analyze a review to reveal which words
            received the most attention.
        </div>
        """

    weights = np.array(weights, dtype=float)

    if len(weights) != len(tokens):
        weights = weights[:len(tokens)]

    if len(tokens) > len(weights):
        tokens = tokens[:len(weights)]

    if len(weights) > 0 and weights.max() > 0:
        weights = weights / weights.max()

    html = '<div class="word-cloud">'

    for token, weight in zip(tokens, weights):

        intensity = int(12 + 75 * weight)

        html += (
            f'<span class="attention-word" '
            f'title="Attention: {weight:.3f}" '
            f'style="background: rgba(251,146,60,{intensity/100:.2f});">'
            f'{token}'
            f'</span>'
        )

    html += "</div>"

    return html


def make_transformer_heatmap(tokens, attention):

    if not attention:
        return None

    attention = np.array(attention)

    if attention.ndim != 4:
        return None

    layer_attention = attention[0]

    num_heads = layer_attention.shape[0]

    max_tokens = min(len(tokens), 30)

    display_tokens = tokens[:max_tokens]

    fig, axes = plt.subplots(
        1,
        num_heads,
        figsize=(4 * num_heads, 4)
    )

    if num_heads == 1:
        axes = [axes]

    for head in range(num_heads):

        matrix = layer_attention[head][
            :max_tokens,
            :max_tokens
        ]

        axes[head].imshow(
            matrix,
            aspect="auto"
        )

        axes[head].set_title(
            f"Head {head + 1}",
            fontsize=12,
            fontweight="bold"
        )

        axes[head].set_xticks(
            range(len(display_tokens))
        )

        axes[head].set_yticks(
            range(len(display_tokens))
        )

        axes[head].set_xticklabels(
            display_tokens,
            rotation=90,
            fontsize=7
        )

        axes[head].set_yticklabels(
            display_tokens,
            fontsize=7
        )

    fig.suptitle(
        "Transformer Multi-Head Attention",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig


def build_model_cards(results):

    if not results:
        return """
        <div class="empty-state">
            Your model predictions will appear here.
        </div>
        """

    cards = ""

    model_icons = {
        "Simple RNN": "🔁",
        "LSTM": "🧠",
        "GRU": "⚡",
        "BiLSTM": "↔️",
        "Attention LSTM": "🔎",
        "Transformer": "🧩",
        "DistilBERT": "🤗"
    }

    model_colors = {
        "Simple RNN": "#6366f1",
        "LSTM": "#8b5cf6",
        "GRU": "#06b6d4",
        "BiLSTM": "#3b82f6",
        "Attention LSTM": "#f97316",
        "Transformer": "#ec4899",
        "DistilBERT": "#10b981"
    }

    for model_name, result in results.items():

        prediction = (
            "Positive"
            if result["prediction"] == 1
            else "Negative"
        )

        confidence = (
            max(result["probabilities"]) * 100
        )

        if prediction == "Positive":
            sentiment_class = "positive"
            sentiment_icon = "😊"
        else:
            sentiment_class = "negative"
            sentiment_icon = "😞"

        icon = model_icons.get(
            model_name,
            "🤖"
        )

        accent = model_colors.get(
            model_name,
            "#6366f1"
        )

        cards += f"""
        <div class="model-card"
             style="--accent:{accent};">

            <div class="model-top">

                <div class="model-name">
                    <span class="model-icon">
                        {icon}
                    </span>

                    {model_name}
                </div>

            </div>

            <div class="sentiment {sentiment_class}">
                {sentiment_icon}
                {prediction}
            </div>

            <div class="confidence-label">
                Confidence
                <strong>{confidence:.2f}%</strong>
            </div>

            <div class="confidence-track">

                <div
                    class="confidence-fill"
                    style="width:{confidence:.1f}%;
                           background:{accent};">
                </div>

            </div>

        </div>
        """

    return f"""
    <div class="model-grid">
        {cards}
    </div>
    """


def analyze_review(review):

    review = review.strip()

    if not review:

        return (
            "⚠️ Please enter a movie review.",
            build_model_cards({}),
            """
            <div class="empty-state">
                🔎 No attention data yet.
            </div>
            """,
            None
        )

    try:

        response = requests.post(
            API_URL,
            json={"review": review},
            timeout=300
        )

        if response.status_code != 200:

            return (
                f"❌ API Error: {response.status_code}",
                build_model_cards({}),
                """
                <div class="empty-state">
                    Attention data unavailable.
                </div>
                """,
                None
            )

        data = response.json()

        results = data["results"]

        model_cards = build_model_cards(
            results
        )

        # ----------------------------------------------------
        # ATTENTION LSTM
        # ----------------------------------------------------

        attention_result = results.get(
            "Attention LSTM",
            {}
        )

        attention_tokens = attention_result.get(
            "tokens",
            []
        )

        attention_weights = attention_result.get(
            "attention_weights",
            []
        )

        attention_html = make_attention_html(
            attention_tokens,
            attention_weights
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        transformer_result = results.get(
            "Transformer",
            {}
        )

        transformer_tokens = transformer_result.get(
            "tokens",
            []
        )

        transformer_attention = transformer_result.get(
            "attention",
            []
        )

        transformer_fig = make_transformer_heatmap(
            transformer_tokens,
            transformer_attention
        )

        return (
            "",
            model_cards,
            attention_html,
            transformer_fig
        )

    except Exception as e:

        return (
            f"❌ Could not connect to API: {e}",
            build_model_cards({}),
            """
            <div class="empty-state">
                Attention data unavailable.
            </div>
            """,
            None
        )


# ============================================================
# CSS — COMPLETE VISUAL REDESIGN
# ============================================================

custom_css = """

/* ============================================================
   GLOBAL
   ============================================================ */

body {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(236,72,153,0.12),
            transparent 30%
        ),
        #f7f8fc;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;

    overflow: hidden;

    padding: 55px 55px 48px;

    border-radius: 30px;

    margin-bottom: 25px;

    color: white;

    background:
        linear-gradient(
            135deg,
            #312e81 0%,
            #4f46e5 35%,
            #7c3aed 68%,
            #db2777 100%
        );

    box-shadow:
        0 25px 60px
        rgba(79,70,229,0.25);
}

.hero:before {
    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.08);

    right: -80px;
    top: -100px;
}

.hero:after {
    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.06);

    left: 45%;
    bottom: -100px;
}

.hero-content {
    position: relative;

    z-index: 2;
}

.hero-kicker {
    text-transform: uppercase;

    letter-spacing: 3px;

    font-size: 12px;

    font-weight: 700;

    opacity: 0.8;

    margin-bottom: 14px;
}

.hero-title {
    font-size: 46px;

    line-height: 1.05;

    font-weight: 900;

    letter-spacing: -1.5px;

    margin-bottom: 16px;
}

.hero-subtitle {
    font-size: 18px;

    line-height: 1.6;

    max-width: 800px;

    color: #ede9fe;
}

.hero-tags {
    margin-top: 26px;
}

.hero-tag {
    display: inline-block;

    padding: 8px 14px;

    margin: 4px 5px 4px 0;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.13);

    border:
        1px solid
        rgba(255,255,255,0.2);

    font-size: 13px;

    backdrop-filter: blur(8px);
}


/* ============================================================
   MAIN INPUT CARD
   ============================================================ */

.input-card {
    background: white;

    border-radius: 24px;

    padding: 28px;

    margin-bottom: 24px;

    border:
        1px solid
        #e5e7eb;

    box-shadow:
        0 12px 35px
        rgba(15,23,42,0.07);
}

.input-heading {
    font-size: 24px;

    font-weight: 800;

    color: #111827;

    margin-bottom: 6px;
}

.input-description {
    color: #6b7280;

    margin-bottom: 18px;
}


/* ============================================================
   TEXT BOX
   ============================================================ */

textarea,
textarea:focus,
textarea:hover {
    background: #ffffff !important;

    color: #111827 !important;

    -webkit-text-fill-color: #111827 !important;

    caret-color: #111827 !important;

    border-radius: 18px !important;

    border: 2px solid #c7d2fe !important;

    font-size: 17px !important;

    line-height: 1.65 !important;

    padding: 18px !important;

    opacity: 1 !important;
}

textarea::placeholder {
    color: #64748b !important;

    -webkit-text-fill-color: #64748b !important;

    opacity: 1 !important;
}


/* ============================================================
   ANALYZE BUTTON
   ============================================================ */

.analyze-btn {
    min-height: 56px !important;

    border-radius: 16px !important;

    font-size: 17px !important;

    font-weight: 800 !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;

    border: none !important;

    box-shadow:
        0 8px 20px
        rgba(79,70,229,0.25);

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

.analyze-btn:hover {
    transform:
        translateY(-2px);

    box-shadow:
        0 12px 28px
        rgba(79,70,229,0.32);
}


/* ============================================================
   SECTION HEADERS
   ============================================================ */

.section-heading {
    font-size: 27px;

    font-weight: 850;

    color: #111827;

    margin-top: 5px;

    margin-bottom: 5px;
}

.section-description {
    color: #6b7280;

    font-size: 15px;

    margin-bottom: 18px;
}


/* ============================================================
   MODEL DASHBOARD
   ============================================================ */

.model-grid {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 16px;

    margin-top: 8px;
}

.model-card {
    position: relative;

    overflow: hidden;

    background: white;

    border:
        1px solid
        #e5e7eb;

    border-radius: 20px;

    padding: 20px;

    min-height: 175px;

    box-shadow:
        0 8px 22px
        rgba(15,23,42,0.055);

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease;
}

.model-card:hover {
    transform:
        translateY(-4px);

    box-shadow:
        0 15px 32px
        rgba(15,23,42,0.11);
}

.model-card:before {
    content: "";

    position: absolute;

    left: 0;
    top: 0;

    width: 5px;
    height: 100%;

    background:
        var(--accent);
}

.model-top {
    display: flex;

    justify-content: space-between;

    align-items: center;
}

.model-name {
    font-size: 15px;

    font-weight: 800;

    color: #374151;
}

.model-icon {
    font-size: 20px;

    margin-right: 7px;
}

.sentiment {
    display: inline-block;

    margin-top: 20px;

    padding: 7px 12px;

    border-radius: 999px;

    font-weight: 800;

    font-size: 15px;
}

.sentiment.positive {
    color: #047857;

    background:
        #d1fae5;
}

.sentiment.negative {
    color: #be123c;

    background:
        #ffe4e6;
}

.confidence-label {
    display: flex;

    justify-content: space-between;

    margin-top: 20px;

    color: #6b7280;

    font-size: 13px;
}

.confidence-label strong {
    color: #111827;
}

.confidence-track {
    height: 7px;

    background:
        #eef2f7;

    border-radius: 999px;

    overflow: hidden;

    margin-top: 8px;
}

.confidence-fill {
    height: 100%;

    border-radius: 999px;
}


/* ============================================================
   EXPLAINABILITY AREA
   ============================================================ */

.explain-card {
    background: white;

    border-radius: 24px;

    padding: 28px;

    margin-top: 24px;

    border:
        1px solid
        #e5e7eb;

    box-shadow:
        0 10px 30px
        rgba(15,23,42,0.06);
}

.attention-container {
    background:
        linear-gradient(
            135deg,
            #fff7ed,
            #fffbeb
        );

    border:
        1px solid
        #fed7aa;

    border-radius: 18px;

    padding: 24px;

    line-height: 2.8;

    min-height: 90px;
}

.attention-word {
    display: inline-block;

    padding: 3px 9px;

    margin: 3px;

    border-radius: 8px;

    font-size: 16px;

    color: #431407;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

.attention-word:hover {
    transform:
        translateY(-3px);

    box-shadow:
        0 5px 14px
        rgba(249,115,22,0.2);
}


/* ============================================================
   TRANSFORMER
   ============================================================ */

.transformer-card {
    background: white;

    border-radius: 24px;

    padding: 28px;

    margin-top: 24px;

    border:
        1px solid
        #e5e7eb;

    box-shadow:
        0 10px 30px
        rgba(15,23,42,0.06);
}


/* ============================================================
   EMPTY STATE
   ============================================================ */

.empty-state {
    background:
        #f8fafc;

    border:
        2px dashed
        #cbd5e1;

    border-radius: 18px;

    padding: 30px;

    text-align: center;

    color: #64748b;

    font-size: 15px;
}


/* ============================================================
   EXAMPLES
   ============================================================ */

.examples-card {
    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #fdf4ff
        );

    border-radius: 24px;

    padding: 26px;

    margin-top: 24px;

    border:
        1px solid
        #ddd6fe;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    padding: 28px 10px;

    color: #94a3b8;

    font-size: 13px;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .model-grid {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .hero-title {
        font-size: 36px;
    }
}

@media (max-width: 600px) {

    .model-grid {
        grid-template-columns:
            1fr;
    }

    .hero {
        padding: 35px 25px;
    }

    .hero-title {
        font-size: 32px;
    }
}

"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="MovieMind — NNDL Sentiment Analyser",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate"
    )
) as demo:


    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-content">

                <div class="hero-kicker">
                    NNDL • DEEP LEARNING PROJECT
                </div>

                <div class="hero-title">
                    🎬 MovieMind
                </div>

                <div class="hero-subtitle">
                    Explore how seven different deep learning
                    architectures understand the sentiment hidden
                    inside a movie review.
                </div>

                <div class="hero-tags">

                    <span class="hero-tag">
                        🔁 RNN
                    </span>

                    <span class="hero-tag">
                        🧠 LSTM
                    </span>

                    <span class="hero-tag">
                        ⚡ GRU
                    </span>

                    <span class="hero-tag">
                        ↔️ BiLSTM
                    </span>

                    <span class="hero-tag">
                        🔎 Attention
                    </span>

                    <span class="hero-tag">
                        🧩 Transformer
                    </span>

                    <span class="hero-tag">
                        🤗 DistilBERT
                    </span>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # INPUT
    # ========================================================

    with gr.Group(elem_classes="input-card"):

        gr.HTML(
            """
            <div class="input-heading">
                ✍️ Tell us about the movie
            </div>

            <div class="input-description">
                Enter a movie review and let the models
                predict whether its sentiment is positive
                or negative.
            </div>
            """
        )

        review = gr.Textbox(
            label="",
            placeholder=(
                "Example: The story was beautifully written, "
                "the performances were amazing, and I loved "
                "every moment of this movie..."
            ),
            lines=6,
            max_lines=10
        )

        analyze_button = gr.Button(
            "✨ Analyze My Review",
            variant="primary",
            elem_classes="analyze-btn"
        )


    # ========================================================
    # ERROR
    # ========================================================

    error_message = gr.Markdown()


    # ========================================================
    # MODEL DASHBOARD
    # ========================================================

    with gr.Group(elem_classes="explain-card"):

        gr.HTML(
            """
            <div class="section-heading">
                📊 Model Dashboard
            </div>

            <div class="section-description">
                Seven independent models analyze the same review.
                Compare their predicted sentiment and confidence.
            </div>
            """
        )

        results_html = gr.HTML(
            """
            <div class="empty-state">
                🎥 Your model results will appear here.
            </div>
            """
        )


    # ========================================================
    # ATTENTION LSTM
    # ========================================================

    with gr.Group(elem_classes="explain-card"):

        gr.HTML(
            """
            <div class="section-heading">
                🔎 Inside the Attention LSTM
            </div>

            <div class="section-description">
                The highlighted words show where the Attention
                LSTM focused when making its prediction.
                Darker highlights represent stronger attention.
            </div>
            """
        )

        attention_output = gr.HTML(
            """
            <div class="empty-state">
                🔎 Analyze a review to reveal word-level attention.
            </div>
            """
        )


    # ========================================================
    # TRANSFORMER
    # ========================================================

    with gr.Group(elem_classes="transformer-card"):

        gr.HTML(
            """
            <div class="section-heading">
                🧩 Transformer Attention
            </div>

            <div class="section-description">
                Explore how the custom Transformer distributes
                attention across different words and attention heads.
            </div>
            """
        )

        transformer_output = gr.Plot(
            label=""
        )


    # ========================================================
    # EXAMPLES
    # ========================================================

    with gr.Group(elem_classes="examples-card"):

        gr.HTML(
            """
            <div class="section-heading">
                🍿 Try a Review
            </div>

            <div class="section-description">
                Click one of the examples below to test the models.
            </div>
            """
        )

        gr.Examples(
            examples=[
                [
                    "This movie was absolutely fantastic and "
                    "I loved every minute of it."
                ],
                [
                    "The movie was terrible and I hated "
                    "every second of it."
                ],
                [
                    "It was okay, but nothing special."
                ],
                [
                    "The acting was brilliant, the story was "
                    "beautiful, and the ending was unforgettable."
                ],
                [
                    "I expected much more from this movie. "
                    "The plot was boring and the characters "
                    "were poorly written."
                ]
            ],
            inputs=review
        )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">
            🎬 MovieMind &nbsp;•&nbsp;
            NNDL Movie Sentiment Analysis
            <br>
            RNN → LSTM → GRU → BiLSTM →
            Attention → Transformer → DistilBERT
        </div>
        """
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_button.click(
        fn=analyze_review,
        inputs=review,
        outputs=[
            error_message,
            results_html,
            attention_output,
            transformer_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch()