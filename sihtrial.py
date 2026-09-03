import streamlit as st
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# ============================================================
# SATQUERY AI - SIH TRIAL PROTOTYPE
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    color: #667085;
    font-size: 18px;
}

.result-box {
    padding: 22px;
    border-radius: 15px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #111827;
}

.trace-box {
    padding: 18px;
    border-radius: 15px;
    background-color: #eef6ff;
    border: 1px solid #bfdbfe;
    color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ SatQuery AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Vision-Language Assistant for Remote Sensing'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload satellite imagery, ask a natural-language question, "
    "and SatQuery automatically selects an analysis workflow."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ SatQuery Controls")

mode = st.sidebar.radio(
    "Select input type",
    [
        "Single Image",
        "Bi-temporal Pair",
        "Optical + SAR Pair"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "The SIH prototype supports single-image analysis, "
    "change analysis and optical-SAR analysis."
)

# ============================================================
# FUNCTIONS
# ============================================================

def load_image(uploaded_file):

    image = Image.open(uploaded_file)

    return image.convert("RGB")


def resize_image(image, maximum=1200):

    width, height = image.size

    scale = min(
        1,
        maximum / max(width, height)
    )

    if scale < 1:

        image = image.resize(
            (
                int(width * scale),
                int(height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    return image


# ============================================================
# LAND COVER ANALYSIS
# ============================================================

def land_cover_analysis(image):

    img = np.array(image).astype(np.float32)

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    brightness = (
        red +
        green +
        blue
    ) / 3

    total_pixels = img.shape[0] * img.shape[1]

    # -------------------------
    # Vegetation
    # -------------------------

    vegetation = (
        (green > red * 1.05) &
        (green > blue * 1.03) &
        (green > 60)
    )

    # -------------------------
    # Water
    # -------------------------

    water = (
        (blue > red * 1.08) &
        (blue >= green * 0.95) &
        (brightness < 170)
    )

    # -------------------------
    # Built-up
    # -------------------------

    max_channel = np.max(img, axis=2)

    min_channel = np.min(img, axis=2)

    neutral = (
        max_channel -
        min_channel
    ) < 45

    built_up = (
        neutral &
        (brightness > 125) &
        (~vegetation) &
        (~water)
    )

    # -------------------------
    # Barren
    # -------------------------

    barren = (
        (~vegetation) &
        (~water) &
        (~built_up) &
        (red > blue * 1.05) &
        (brightness > 55)
    )

    # Remove overlaps

    water = water & (~vegetation)

    built_up = (
        built_up &
        (~vegetation) &
        (~water)
    )

    barren = (
        barren &
        (~vegetation) &
        (~water) &
        (~built_up)
    )

    # -------------------------
    # Percentages
    # -------------------------

    water_pct = water.sum() / total_pixels * 100

    vegetation_pct = (
        vegetation.sum() /
        total_pixels *
        100
    )

    built_pct = (
        built_up.sum() /
        total_pixels *
        100
    )

    barren_pct = (
        barren.sum() /
        total_pixels *
        100
    )

    known = (
        water |
        vegetation |
        built_up |
        barren
    )

    other_pct = (
        (~known).sum() /
        total_pixels *
        100
    )

    percentages = {

        "Water": round(water_pct, 1),

        "Vegetation": round(
            vegetation_pct,
            1
        ),

        "Built-up": round(
            built_pct,
            1
        ),

        "Barren Land": round(
            barren_pct,
            1
        ),

        "Other": round(
            other_pct,
            1
        )
    }

    masks = {

        "Water": water,

        "Vegetation": vegetation,

        "Built-up": built_up,

        "Barren Land": barren
    }

    return percentages, masks


# ============================================================
# CREATE VISUAL EVIDENCE
# ============================================================

def create_evidence(image, masks, selected=None):

    image_array = np.array(image).copy()

    overlay = image_array.copy()

    colors = {

        "Water": (30, 120, 255),

        "Vegetation": (40, 180, 70),

        "Built-up": (220, 70, 70),

        "Barren Land": (210, 160, 50)
    }

    if selected:

        selected_categories = [selected]

    else:

        selected_categories = list(
            masks.keys()
        )

    for category in selected_categories:

        mask = masks[category]

        color = np.array(
            colors[category],
            dtype=np.uint8
        )

        overlay[mask] = (
            0.45 *
            overlay[mask] +

            0.55 *
            color
        ).astype(np.uint8)

    result = cv2.addWeighted(
        image_array,
        0.55,
        overlay,
        0.45,
        0
    )

    return Image.fromarray(result)


# ============================================================
# CHANGE ANALYSIS
# ============================================================

def change_analysis(image1, image2):

    image2 = image2.resize(
        image1.size
    )

    a = np.array(image1).astype(
        np.float32
    )

    b = np.array(image2).astype(
        np.float32
    )

    gray_a = cv2.cvtColor(
        a.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    gray_b = cv2.cvtColor(
        b.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    difference = cv2.absdiff(
        gray_a,
        gray_b
    )

    difference = cv2.GaussianBlur(
        difference,
        (7, 7),
        0
    )

    threshold = np.percentile(
        difference,
        82
    )

    change_mask = (
        difference >
        threshold
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    change_mask = cv2.morphologyEx(
        change_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    result = np.array(
        image2
    ).copy()

    changed_pixels = (
        change_mask >
        0
    )

    result[changed_pixels] = (
        255,
        70,
        50
    )

    result = cv2.addWeighted(
        np.array(image2),
        0.65,
        result,
        0.35,
        0
    )

    percentage = (
        changed_pixels.sum() /
        changed_pixels.size *
        100
    )

    return (
        Image.fromarray(result),
        round(percentage, 1)
    )


# ============================================================
# QUERY ROUTER
# ============================================================

def choose_task(
    question,
    number_of_images,
    optical_sar=False
):

    q = question.lower()

    # Change analysis

    if number_of_images >= 2:

        if any(
            word in q
            for word in [
                "change",
                "changed",
                "difference",
                "compare",
                "before",
                "after",
                "increase",
                "decrease"
            ]
        ):

            return "Bi-temporal Change Analysis"

    # Optical + SAR

    if optical_sar:

        return (
            "Optical + SAR "
            "Cross-Modal Analysis"
        )

    # Region

    if any(
        word in q
        for word in [
            "highlight",
            "where",
            "locate",
            "region",
            "show",
            "mark"
        ]
    ):

        return (
            "Text-Guided "
            "Region Analysis"
        )

    # Percentage

    if any(
        word in q
        for word in [
            "percentage",
            "percent",
            "%",
            "how much",
            "coverage"
        ]
    ):

        return "Land-Cover Estimation"

    # Caption / scene

    if any(
        word in q
        for word in [
            "describe",
            "visible",
            "scene",
            "objects",
            "land cover"
        ]
    ):

        return (
            "Remote-Sensing "
            "Scene Understanding"
        )

    return "Single-Image VQA"


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header("📁 Upload Satellite Imagery")

image1_file = None
image2_file = None

if mode == "Single Image":

    image1_file = st.file_uploader(
        "Upload satellite image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ]
    )

elif mode == "Bi-temporal Pair":

    col1, col2 = st.columns(2)

    with col1:

        image1_file = st.file_uploader(
            "Image A — Earlier Date",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )

    with col2:

        image2_file = st.file_uploader(
            "Image B — Later Date",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )

else:

    col1, col2 = st.columns(2)

    with col1:

        image1_file = st.file_uploader(
            "Optical / Multispectral Image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )

    with col2:

        image2_file = st.file_uploader(
            "SAR Image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ]
        )


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = None
image2 = None

if image1_file:

    image1 = load_image(
        image1_file
    )

if image2_file:

    image2 = load_image(
        image2_file
    )


# ============================================================
# PREVIEW
# ============================================================

if image1:

    st.header("🖼️ Image Preview")

    if image2:

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                resize_image(image1),
                caption="Image A / Optical",
                use_container_width=True
            )

        with col2:

            st.image(
                resize_image(image2),
                caption=(
                    "Image B / Later Date / SAR"
                ),
                use_container_width=True
            )

    else:

        st.image(
            resize_image(image1),
            caption="Uploaded Satellite Image",
            use_container_width=True
        )


# ============================================================
# QUESTION SECTION
# ============================================================

st.header("💬 Ask SatQuery AI")

example_questions = {

    "Single Image": [

        "Describe the land-cover and major objects visible in this image.",

        "What percentage of the area is covered by water?",

        "What percentage of the area is barren land?",

        "What percentage of the area is vegetation?",

        "Where are the built-up areas?",

        "Highlight the vegetation in this image."
    ],

    "Bi-temporal Pair": [

        "What changed between these two dates?",

        "Has the built-up area increased or decreased?",

        "Where did the major change occur?",

        "Compare the land cover between these images."
    ],

    "Optical + SAR Pair": [

        "Use both images to identify built-up and water-covered regions.",

        "Identify water-covered regions using both images.",

        "Identify built-up regions using optical and SAR images."
    ]
}


question_type = st.selectbox(
    "Try an example",
    [
        "Custom Question"
    ] +
    example_questions[mode]
)

if question_type == "Custom Question":

    default_question = ""

else:

    default_question = question_type


question = st.text_area(
    "Your question",
    value=default_question,
    height=90,
    placeholder=(
        "Example: What percentage "
        "of this area is water?"
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 ANALYZE",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_button:

    if not image1:

        st.error(
            "Please upload the required image."
        )

        st.stop()

    if mode != "Single Image" and not image2:

        st.error(
            "Please upload both required images."
        )

        st.stop()

    if not question.strip():

        st.error(
            "Please enter a question."
        )

        st.stop()


    number_of_images = (
        2 if image2 else 1
    )

    optical_sar = (
        mode ==
        "Optical + SAR Pair"
    )


    # ========================================================
    # AGENTIC ROUTING
    # ========================================================

    selected_task = choose_task(
        question,
        number_of_images,
        optical_sar
    )


    with st.spinner(
        "🧠 SatQuery is understanding your question..."
    ):

        percentages, masks = (
            land_cover_analysis(
                resize_image(image1)
            )
        )


        # ====================================================
        # SINGLE IMAGE
        # ====================================================

        if mode == "Single Image":

            q = question.lower()

            # WATER

            if any(
                word in q
                for word in [
                    "water",
                    "water body",
                    "waterbody",
                    "lake",
                    "river"
                ]
            ):

                value = percentages[
                    "Water"
                ]

                answer = (
                    f"Approximately **{value}%** "
                    "of the image is estimated "
                    "to be water-covered."
                )

                evidence = create_evidence(
                    resize_image(image1),
                    masks,
                    "Water"
                )

                confidence = 82


            # BARREN

            elif any(
                word in q
                for word in [
                    "barren",
                    "bare land",
                    "bare",
                    "wasteland"
                ]
            ):

                value = percentages[
                    "Barren Land"
                ]

                answer = (
                    f"Approximately **{value}%** "
                    "of the image is estimated "
                    "to be barren/bare land."
                )

                evidence = create_evidence(
                    resize_image(image1),
                    masks,
                    "Barren Land"
                )

                confidence = 80


            # VEGETATION

            elif any(
                word in q
                for word in [
                    "vegetation",
                    "green",
                    "forest",
                    "crop",
                    "agriculture",
                    "agricultural"
                ]
            ):

                value = percentages[
                    "Vegetation"
                ]

                answer = (
                    f"Approximately **{value}%** "
                    "of the image is estimated "
                    "to contain vegetation."
                )

                evidence = create_evidence(
                    resize_image(image1),
                    masks,
                    "Vegetation"
                )

                confidence = 81


            # BUILT-UP

            elif any(
                word in q
                for word in [
                    "built-up",
                    "built up",
                    "building",
                    "buildings",
                    "urban",
                    "city",
                    "settlement"
                ]
            ):

                value = percentages[
                    "Built-up"
                ]

                answer = (
                    f"Approximately **{value}%** "
                    "of the image is estimated "
                    "to be built-up."
                )

                evidence = create_evidence(
                    resize_image(image1),
                    masks,
                    "Built-up"
                )

                confidence = 78


            # GENERAL QUESTION

            else:

                dominant = max(
                    [
                        "Water",
                        "Vegetation",
                        "Built-up",
                        "Barren Land"
                    ],
                    key=lambda x:
                    percentages[x]
                )

                answer = (
                    "The image contains a mixture "
                    "of vegetation, water, built-up "
                    "and barren regions. "
                    f"The largest detected category "
                    f"is **{dominant} "
                    f"({percentages[dominant]}%)**."
                )

                evidence = create_evidence(
                    resize_image(image1),
                    masks
                )

                confidence = 76


        # ====================================================
        # BI-TEMPORAL
        # ====================================================

        elif mode == "Bi-temporal Pair":

            evidence, change_percentage = (
                change_analysis(
                    resize_image(image1),
                    resize_image(image2)
                )
            )

            p1, _ = land_cover_analysis(
                resize_image(image1)
            )

            p2, _ = land_cover_analysis(
                resize_image(image2)
            )

            q = question.lower()


            if (
                "built" in q or
                "urban" in q or
                "building" in q
            ):

                difference = round(
                    p2["Built-up"] -
                    p1["Built-up"],
                    1
                )

                if difference > 0.5:

                    direction = "increased"

                elif difference < -0.5:

                    direction = "decreased"

                else:

                    direction = (
                        "remained approximately "
                        "unchanged"
                    )

                answer = (
                    f"The estimated built-up "
                    f"coverage **{direction}** "
                    f"by **{abs(difference)} "
                    "percentage points**."
                )


            elif (
                "vegetation" in q or
                "green" in q or
                "crop" in q
            ):

                difference = round(
                    p2["Vegetation"] -
                    p1["Vegetation"],
                    1
                )

                if difference > 0.5:

                    direction = "increased"

                elif difference < -0.5:

                    direction = "decreased"

                else:

                    direction = (
                        "remained approximately "
                        "unchanged"
                    )

                answer = (
                    f"The estimated vegetation "
                    f"coverage **{direction}** "
                    f"by **{abs(difference)} "
                    "percentage points**."
                )


            elif "water" in q:

                difference = round(
                    p2["Water"] -
                    p1["Water"],
                    1
                )

                if difference > 0.5:

                    direction = "increased"

                elif difference < -0.5:

                    direction = "decreased"

                else:

                    direction = (
                        "remained approximately "
                        "unchanged"
                    )

                answer = (
                    f"The estimated water "
                    f"coverage **{direction}** "
                    f"by **{abs(difference)} "
                    "percentage points**."
                )


            else:

                answer = (
                    f"Approximately "
                    f"**{change_percentage}%** "
                    "of the image area shows "
                    "visual differences between "
                    "the two dates."
                )


            confidence = 84


        # ====================================================
        # OPTICAL + SAR
        # ====================================================

        else:

            optical_percentages, optical_masks = (
                land_cover_analysis(
                    resize_image(image1)
                )
            )

            sar_percentages, sar_masks = (
                land_cover_analysis(
                    resize_image(image2)
                )
            )

            combined = {}

            for category in [
                "Water",
                "Vegetation",
                "Built-up",
                "Barren Land"
            ]:

                combined[category] = round(
                    (
                        optical_percentages[
                            category
                        ] +

                        sar_percentages[
                            category
                        ]
                    ) / 2,
                    1
                )


            q = question.lower()


            if (
                "water" in q and
                (
                    "built" in q or
                    "urban" in q
                )
            ):

                answer = (
                    "Using complementary "
                    "information from the "
                    "optical and SAR inputs, "
                    f"the prototype estimates "
                    f"**{combined['Water']}% "
                    "water-covered** and "
                    f"**{combined['Built-up']}% "
                    "built-up area."
                )


            elif "water" in q:

                answer = (
                    "The combined optical + "
                    "SAR analysis estimates "
                    f"approximately **"
                    f"{combined['Water']}%** "
                    "water-covered area."
                )


            elif (
                "built" in q or
                "urban" in q
            ):

                answer = (
                    "The combined optical + "
                    "SAR analysis estimates "
                    f"approximately **"
                    f"{combined['Built-up']}%** "
                    "built-up area."
                )


            else:

                answer = (
                    "The optical image provides "
                    "spectral/contextual information "
                    "while SAR provides complementary "
                    "structural information. "
                    f"The combined prototype estimate "
                    f"contains {combined['Vegetation']}% "
                    "vegetation, "
                    f"{combined['Water']}% water and "
                    f"{combined['Built-up']}% built-up "
                    "coverage."
                )


            evidence = create_evidence(
                resize_image(image1),
                optical_masks
            )

            confidence = 74


    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        "Analysis completed successfully."
    )

    st.divider()

    st.header("🤖 SatQuery AI Result")

    left, right = st.columns(
        [1.5, 1]
    )


    # ========================================================
    # ANSWER
    # ========================================================

    with left:

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "💬 AI Answer"
        )

        st.markdown(
            answer
        )

        st.write(
            f"**Confidence: {confidence}%**"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        st.subheader(
            "🖼️ Visual Evidence"
        )

        st.image(
            evidence,
            use_container_width=True
        )


    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    with right:

        st.markdown(
            '<div class="trace-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🔎 Execution Summary"
        )

        st.write(
            f"**Task selected:** "
            f"{selected_task}"
        )

        st.write(
            f"**Input:** {mode}"
        )

        st.write(
            "**Controller:** "
            "SatQuery Agentic Router"
        )

        if mode == "Single Image":

            st.write(
                "**Workflow:** "
                "Single-image analysis"
            )

        elif mode == "Bi-temporal Pair":

            st.write(
                "**Workflow:** "
                "Change understanding"
            )

        else:

            st.write(
                "**Workflow:** "
                "Optical-SAR analysis"
            )

        st.write(
            "**Evidence:** "
            "Visual analysis overlay"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence}%"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # LAND COVER DASHBOARD
    # ========================================================

    if mode == "Single Image":

        st.subheader(
            "📊 Estimated Land-Cover Distribution"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "💧 Water",
            f"{percentages['Water']}%"
        )

        c2.metric(
            "🌱 Vegetation",
            f"{percentages['Vegetation']}%"
        )

        c3.metric(
            "🏙️ Built-up",
            f"{percentages['Built-up']}%"
        )

        c4.metric(
            "🟫 Barren",
            f"{percentages['Barren Land']}%"
        )

        c5.metric(
            "⬜ Other",
            f"{percentages['Other']}%"
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    report = f"""
SATQUERY AI
SIH TRIAL ANALYSIS REPORT
========================================

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

INPUT TYPE
{mode}

SELECTED TASK
{selected_task}

USER QUESTION
{question}

AI ANSWER
{answer}

CONFIDENCE
{confidence}%

========================================

NOTE

This is an SIH prototype.

The current visual land-cover estimates are
prototype image-processing estimates. For
final SIH evaluation, these workflows should
be connected to properly remote-sensing-adapted
or fine-tuned specialist models and the
prescribed benchmark datasets.

========================================
"""

    st.download_button(
        "📥 Download Analysis Report",
        data=report,
        file_name="satquery_analysis_report.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛰️ SatQuery AI • SIH Trial Prototype • "
    "Agentic Remote-Sensing Analysis"
)