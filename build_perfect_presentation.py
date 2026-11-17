import os
import fitz  # PyMuPDF
from PIL import Image, ImageChops
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# -------------------------------------------------------------
# 1. IMAGE EXTRACTION & CLEAN CROPPING
# -------------------------------------------------------------
def extract_and_crop_images(pdf_path):
    print("Extracting and cropping images from PDF...")
    doc = fitz.open(pdf_path)
    zoom = 300 / 72  # 300 DPI
    mat = fitz.Matrix(zoom, zoom)

    images = {}

    def save_clean_crop(pix, out_path, padding=8):
        temp_path = out_path + ".temp.png"
        pix.save(temp_path)
        im = Image.open(temp_path).convert('RGB')
        bg = Image.new(im.mode, im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if bbox:
            l = max(0, bbox[0] - padding)
            t = max(0, bbox[1] - padding)
            r = min(im.size[0], bbox[2] + padding)
            b = min(im.size[1], bbox[3] + padding)
            cropped = im.crop((l, t, r, b))
            cropped.save(out_path)
        else:
            im.save(out_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 1. Figure 1: Rationale win rates (Page 2, index 1)
    page2 = doc[1]
    pix1 = page2.get_pixmap(matrix=mat, clip=fitz.Rect(68, 125, 295, 268))
    save_clean_crop(pix1, "fig1_clean.png", padding=8)
    images["fig1"] = "fig1_clean.png"

    # 2. Figure 2: Architecture diagram (Page 4, index 3)
    page4 = doc[3]
    pix2 = page4.get_pixmap(matrix=mat, clip=fitz.Rect(68, 65, 528, 258))
    save_clean_crop(pix2, "fig2_clean.png", padding=6)
    images["fig2"] = "fig2_clean.png"

    # 3. Table 1: Supervised Fine-Tuning results (Page 6, index 5)
    page6 = doc[5]
    pix3 = page6.get_pixmap(matrix=mat, clip=fitz.Rect(68, 68, 528, 410))
    save_clean_crop(pix3, "table1_clean.png", padding=8)
    images["table1"] = "table1_clean.png"

    # 4. Table 2: Out-of-Domain results (Page 7, index 6)
    page7 = doc[6]
    pix4 = page7.get_pixmap(matrix=mat, clip=fitz.Rect(68, 68, 528, 335))
    save_clean_crop(pix4, "table2_clean.png", padding=8)
    images["table2"] = "table2_clean.png"

    print("Images extracted and cleanly cropped successfully.")
    return images


# -------------------------------------------------------------
# 2. THEME & GEOMETRY CONSTANTS
# -------------------------------------------------------------
COLOR_BG = RGBColor(248, 250, 252)         # #F8FAFC
COLOR_HEADER_BG = RGBColor(20, 45, 80)     # #142D50 (Deep Cambridge Navy)
COLOR_ACCENT = RGBColor(2, 132, 199)       # #0284C7 (Azure Blue Accent)
COLOR_TITLE_TEXT = RGBColor(255, 255, 255) # White
COLOR_HEADING = RGBColor(15, 23, 42)       # #0F172A (Deep Slate)
COLOR_BODY = RGBColor(51, 65, 85)          # #334155 (Charcoal Slate)
COLOR_MUTED = RGBColor(100, 116, 139)      # #64748B (Slate Muted)
COLOR_BORDER = RGBColor(226, 232, 240)     # Light Border


def bring_to_front(slide, shape):
    """Brings a shape to the very front of the visual display z-order."""
    spTree = slide.shapes._spTree
    elem = shape._element
    spTree.remove(elem)
    spTree.append(elem)


def send_to_back(slide, shape):
    """Sends a shape behind other shapes."""
    spTree = slide.shapes._spTree
    elem = shape._element
    spTree.remove(elem)
    spTree.insert(2, elem)


def apply_slide_theme(slide, slide_num, total_slides=14, is_title_slide=False):
    # Background fill
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG

    if not is_title_slide:
        # Top Header Banner
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.1))
        header.fill.solid()
        header.fill.fore_color.rgb = COLOR_HEADER_BG
        header.line.fill.background()
        send_to_back(slide, header)

        # Accent Stripe beneath Header
        stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.1), Inches(10), Inches(0.05))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = COLOR_ACCENT
        stripe.line.fill.background()
        send_to_back(slide, stripe)

        # Bottom Footer Divider
        footer_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.95), Inches(8.4), Inches(0.015))
        footer_line.fill.solid()
        footer_line.fill.fore_color.rgb = COLOR_BORDER
        footer_line.line.fill.background()

        # Footer Text
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.02), Inches(7.0), Inches(0.35))
        tf_f = footer_box.text_frame
        tf_f.word_wrap = True
        tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
        p_f = tf_f.paragraphs[0]
        p_f.text = "Robust Adaptation of LMMs for Retrieval Augmented Hateful Meme Detection | Cambridge (EMNLP 2025)"
        p_f.font.size = Pt(9)
        p_f.font.color.rgb = COLOR_MUTED
        p_f.font.name = "Calibri"

        # Slide Number
        num_box = slide.shapes.add_textbox(Inches(8.0), Inches(7.02), Inches(1.2), Inches(0.35))
        tf_n = num_box.text_frame
        tf_n.word_wrap = True
        tf_n.margin_left = tf_n.margin_top = tf_n.margin_right = tf_n.margin_bottom = 0
        p_n = tf_n.paragraphs[0]
        p_n.text = f"{slide_num} / {total_slides}"
        p_n.alignment = PP_ALIGN.RIGHT
        p_n.font.size = Pt(9.5)
        p_n.font.bold = True
        p_n.font.color.rgb = COLOR_MUTED
        p_n.font.name = "Calibri"


def set_slide_title(slide, title_text):
    """Configures the slide's official title placeholder and ensures it sits in front of the banner."""
    title_shape = slide.shapes.title
    bring_to_front(slide, title_shape)
    title_shape.left = Inches(0.8)
    title_shape.top = Inches(0.18)
    title_shape.width = Inches(8.4)
    title_shape.height = Inches(0.75)
    
    tf = title_shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(25)
    p.font.bold = True
    p.font.color.rgb = COLOR_TITLE_TEXT
    p.font.name = "Calibri"


def create_content_textbox(slide, left, top, width, height):
    """Creates a text box with standard margins and word wrapping."""
    tx_box = slide.shapes.add_textbox(left, top, width, height)
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_top = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    return tf


def add_bullet_point(tf, heading, text, is_first=False, pt_heading=15, pt_body=13.5, space_after=6):
    """Adds a humanized bullet point: bold lead-in heading + clear plain-English body."""
    p = tf.paragraphs[0] if is_first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.line_spacing = 1.15

    # Bold lead-in
    run_h = p.add_run()
    run_h.text = "• " + heading + (" " if heading else "")
    run_h.font.bold = True
    run_h.font.size = Pt(pt_heading)
    run_h.font.color.rgb = COLOR_HEADING
    run_h.font.name = "Calibri"

    # Regular body
    if text:
        run_b = p.add_run()
        run_b.text = text
        run_b.font.bold = False
        run_b.font.size = Pt(pt_body)
        run_b.font.color.rgb = COLOR_BODY
        run_b.font.name = "Calibri"


def add_sub_bullet(tf, text, pt_text=13, space_after=4):
    """Adds an indented secondary detail point in friendly plain English."""
    p = tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.line_spacing = 1.15

    run_indent = p.add_run()
    run_indent.text = "    – " + text
    run_indent.font.size = Pt(pt_text)
    run_indent.font.color.rgb = COLOR_MUTED
    run_indent.font.name = "Calibri"


# -------------------------------------------------------------
# 3. BUILD ALL 14 SLIDES (WITH CITATIONS & BASE PAPER SLIDE)
# -------------------------------------------------------------
def build_presentation(images):
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # =========================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    apply_slide_theme(slide1, 1, 14, is_title_slide=True)

    # Main Card
    card = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.15), Inches(8.4), Inches(5.2))
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.5)

    # Left colored accent line on card
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.15), Inches(0.18), Inches(5.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT
    bar.line.fill.background()

    # Category Pill
    pill = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3), Inches(1.55), Inches(3.2), Inches(0.42))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(238, 246, 255)
    pill.line.color.rgb = RGBColor(186, 224, 255)
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = "RESEARCH PRESENTATION"
    p_pill.font.size = Pt(11)
    p_pill.font.bold = True
    p_pill.font.color.rgb = COLOR_ACCENT
    p_pill.font.name = "Calibri"
    p_pill.alignment = PP_ALIGN.CENTER

    # Main Title
    title_box = slide1.shapes.title
    title_box.left = Inches(1.3)
    title_box.top = Inches(2.2)
    title_box.width = Inches(7.5)
    title_box.height = Inches(1.8)
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p_t = tf1.paragraphs[0]
    p_t.text = "Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection"
    p_t.font.size = Pt(28)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_HEADER_BG
    p_t.font.name = "Calibri"

    # Subtitle / Authors / Conference / Base Paper Citation
    subtitle_box = slide1.placeholders[1]
    subtitle_box.left = Inches(1.3)
    subtitle_box.top = Inches(4.3)
    subtitle_box.width = Inches(7.5)
    subtitle_box.height = Inches(1.6)
    tf_sub = subtitle_box.text_frame
    tf_sub.word_wrap = True
    p_s1 = tf_sub.paragraphs[0]
    p_s1.text = "Authors: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne"
    p_s1.font.size = Pt(15)
    p_s1.font.bold = True
    p_s1.font.color.rgb = COLOR_HEADING
    p_s1.font.name = "Calibri"
    p_s1.space_after = Pt(4)

    p_s2 = tf_sub.add_paragraph()
    p_s2.text = "Department of Engineering, University of Cambridge  |  EMNLP 2025 (Oral Presentation)"
    p_s2.font.size = Pt(14)
    p_s2.font.bold = False
    p_s2.font.color.rgb = COLOR_ACCENT
    p_s2.font.name = "Calibri"
    p_s2.space_after = Pt(4)

    p_s3 = tf_sub.add_paragraph()
    p_s3.text = "Base Paper: arXiv:2502.13061 [cs.CL]  |  Official Code: github.com/JingbiaoMei/RGCL"
    p_s3.font.size = Pt(12)
    p_s3.font.color.rgb = COLOR_MUTED
    p_s3.font.name = "Calibri"

    # Re-order shapes on Slide 1 so card is back, bar is over card, text is over both
    spTree1 = slide1.shapes._spTree
    for s in [card, bar, pill, title_box, subtitle_box]:
        spTree1.remove(s._element)
        spTree1.append(s._element)


    # =========================================================
    # SLIDE 2: THE REAL-WORLD PROBLEM
    # =========================================================
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide2, 2, 14)
    set_slide_title(slide2, "The Problem: Why Hateful Memes Are So Hard to Catch")

    tf2 = create_content_textbox(slide2, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf2, "Everyday Humor Turned Toxic [Kiela et al., NeurIPS 2020]:", 
                     "Memes are how people joke and connect on social media every day. Unfortunately, trolls and bad actors weaponize them to spread hate speech, racism, misogyny, and harassment at massive scale.", is_first=True)
    add_sub_bullet(tf2, "Major platforms like Meta, Reddit, and X get flooded with hundreds of thousands of new memes daily, creating an overwhelming workload for human reviewers [Liu et al., 2022].")

    add_bullet_point(tf2, "The Hidden Meaning Trap [Prakash et al., 2023; Aggarwal et al., WWW 2023]:", 
                     "In a hateful meme, neither the image by itself nor the text by itself looks bad. The hate only appears when you look at both of them together.")
    add_sub_bullet(tf2, "For example, a picture of a peaceful desert looks completely harmless, and the words 'Look at this crowd' look harmless too. But put them together to mock refugees, and it turns into hateful abuse.")
    add_sub_bullet(tf2, "Normal AI filters only check the image OR the text separately, so they completely miss the insult hiding in plain sight.")

    add_bullet_point(tf2, "Sarcasm, Pop Culture & Fast-Moving Slang [Mei et al., ACL 2024]:", 
                     "Memes rely on sarcasm, irony, inside jokes, and rapidly changing internet slang that evolve far faster than static keyword blocklists.")
    add_sub_bullet(tf2, "To catch these, an AI cannot just follow rigid rules; it needs real cultural context, common sense, and visual understanding.")


    # =========================================================
    # SLIDE 3: THE AI DILEMMA
    # =========================================================
    slide3 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide3, 3, 14)
    set_slide_title(slide3, "The AI Dilemma: Why Current Multimodal AI Struggles")

    tf3 = create_content_textbox(slide3, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf3, "1. Giant AI Models Actually Score Worse [Mei et al., 2024]:", 
                     "You would expect huge multi-billion parameter AI models (like LLaVA or Qwen-VL) to excel here. In reality, they often do worse on memes than older, smaller models built specifically for this single job like CLIP [Radford et al., ICML 2021].", is_first=True)
    add_sub_bullet(tf3, "Giant models know a little bit about everything, but they lack the sharp eye needed to spot sly, nuanced internet jokes.")

    add_bullet_point(tf3, "2. Completely Helpless Against New Viral Trends [Huang et al., 2024]:", 
                     "Internet meme culture mutates overnight around breaking news, elections, and viral events [Cao et al., WWW 2024].")
    add_sub_bullet(tf3, "When standard AI is only trained on old meme datasets, it basically memorizes the past. The moment a new viral meme template pops up online, the AI gets totally confused and fails.")

    add_bullet_point(tf3, "3. The AI 'Loses Its Mind' - Catastrophic Forgetting [Yue et al., 2024]:", 
                     "When engineers force an AI to learn memes using standard Supervised Fine-Tuning (SFT), the model suffers from 'catastrophic forgetting'—it forgets how to do normal everyday tasks!")
    add_sub_bullet(tf3, "Its scores on standard benchmark tests like MMMU [Yue et al., ICLR 2024] and GQA [Hudson & Manning, CVPR 2019] drop sharply, ruining the very reason we wanted a smart foundation model in the first place.")


    # =========================================================
    # SLIDE 4: PROPOSED SOLUTION (RA-HMD)
    # =========================================================
    slide4 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide4, 4, 14)
    set_slide_title(slide4, "The Proposed Solution: The RA-HMD Framework")

    tf4 = create_content_textbox(slide4, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf4, "Introducing RA-HMD [Mei, Chen, Yang, Lin, Byrne - EMNLP 2025]:", 
                     "The authors developed RA-HMD (Retrieval-Augmented Hateful Meme Detection). It gives the AI a dedicated helper for decisions and a searchable memory bank of past memes so it doesn't get overwhelmed.", is_first=True)

    add_bullet_point(tf4, "Splitting the Work - Dedicated Judge vs Voice [Kumar & Nandakumar, 2022]:", 
                     "Instead of forcing the AI's speaking voice (LM Head) to make harsh yes/no classifications (which damages general reasoning [Lin et al., 2024]), RA-HMD adds a dedicated classification judge (LRC).")
    add_sub_bullet(tf4, "This keeps the AI's natural conversational voice completely safe while the specialized judge handles the tough call of deciding whether a meme is hateful.")

    add_bullet_point(tf4, "Smart 2-Step Learning + Instant Memory Lookup [Hu et al., 2022; Schroff et al., 2015]:", 
                     "First, it learns how memes work without breaking its normal speaking skills (LoRA). Second, it organizes similar memes together in an orderly digital filing cabinet.")
    add_sub_bullet(tf4, "When a brand-new meme arrives, it instantly searches this filing cabinet for similar past examples to make the right call.")

    add_bullet_point(tf4, "Key Accomplishments [Mei et al., 2025]:", 
                     "Sets all-time high accuracy scores across 6 major datasets, protects over 99.4% of the AI's general intelligence, and writes clear explanations that humans genuinely trust.")


    # =========================================================
    # SLIDE 5: ARCHITECTURE
    # =========================================================
    slide5 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide5, 5, 14)
    set_slide_title(slide5, "System Architecture: How RA-HMD Works Step-by-Step")

    # Add Figure 2 cleanly cropped (aspect ratio ~2.44)
    fig2_width = Inches(7.6)
    fig2_height = Inches(2.9)
    slide5.shapes.add_picture(images["fig2"], Inches(1.2), Inches(1.3), width=fig2_width, height=fig2_height)

    # Explanatory text below the diagram
    tf5 = create_content_textbox(slide5, Inches(0.8), Inches(4.35), Inches(8.4), Inches(2.45))
    add_bullet_point(tf5, "Main AI Brain & Voice [Qwen2-VL-7B; Wang et al., 2024]:", 
                     "Looks at the picture and reads the text together. It keeps practicing normal English so it never forgets how to explain its reasoning in full sentences.", is_first=True, pt_heading=14, pt_body=12.5, space_after=4)
    add_bullet_point(tf5, "The Translator Layer [Projection MLP; Mei et al., 2025]:", 
                     "Takes what the AI sees (hidden state h_i) and turns it into a compact digital summary (vector embedding g_i) that captures the true meaning and intent of the meme.", pt_heading=14, pt_body=12.5, space_after=4)
    add_bullet_point(tf5, "Dedicated Scoring Judge [LRC; Kumar & Nandakumar, 2022]:", 
                     "A specialized classifier operating on g_i answering: 'Is this meme hateful or safe?' This protects the rest of the AI from getting confused.", pt_heading=14, pt_body=12.5, space_after=4)
    add_bullet_point(tf5, "The Meme Memory Bank [FAISS Index; Johnson et al., IEEE 2021]:", 
                     "A searchable library of known memes. When a new meme arrives, the AI compares it against this database to see how similar past memes were handled.", pt_heading=14, pt_body=12.5, space_after=2)


    # =========================================================
    # SLIDE 6: STAGE 1 TRAINING
    # =========================================================
    slide6 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide6, 6, 14)
    set_slide_title(slide6, "Training Step 1: Learning Memes Without Forgetting English")

    tf6 = create_content_textbox(slide6, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf6, "The Goal of Step 1 - Learning Without Forgetting [Mei et al., 2025]:", 
                     "Teach the big AI model how hateful memes work, while making sure it doesn't lose its everyday speaking, reading, and reasoning skills in the process.", is_first=True, pt_heading=15, pt_body=13.5, space_after=5)

    add_bullet_point(tf6, "Lightweight Training with LoRA [Hu et al., ICLR 2022]:", 
                     "Instead of rewiring all 7 billion brain connections (which takes massive supercomputers and risks breaking the AI), it only tunes tiny adapter layers called LoRA.", pt_heading=15, pt_body=13.5, space_after=3)
    add_sub_bullet(tf6, "This makes training fast, uses far less computer memory, and keeps the AI's core foundation completely stable.", pt_text=12.5, space_after=4)

    add_bullet_point(tf6, "Practicing Two Skills at Once - Dual Loss [Lin et al., 2024]:", 
                     "The AI trains on two distinct goals at the exact same time (L_Stage1 = L_LM + L_LR) so neither gets left behind:", pt_heading=15, pt_body=13.5, space_after=3)
    add_sub_bullet(tf6, "1. Normal Language Practice (L_LM): Keeps writing out full sentences so grammar, fluency, and conversational reasoning stay sharp.", pt_text=12.5, space_after=3)
    add_sub_bullet(tf6, "2. Hate Scoring Practice (L_LR): Trains the dedicated judge to accurately tell hateful memes apart from safe ones using cross-entropy.", pt_text=12.5, space_after=4)

    add_bullet_point(tf6, "Why Standard Training Fails Here [Mei et al., 2024]:", 
                     "Standard training forces the AI to only spit out a single word like 'hateful' or 'benign', which dumbs down its internal thinking over time.", pt_heading=15, pt_body=13.5, space_after=3)
    add_sub_bullet(tf6, "Our dual approach sharpens the AI's judgment without taking away its ability to explain its thoughts like a human.", pt_text=12.5, space_after=2)


    # =========================================================
    # SLIDE 7: STAGE 2 TRAINING
    # =========================================================
    slide7 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide7, 7, 14)
    set_slide_title(slide7, "Training Step 2: Sorting Memes by Meaning, Not Appearance")

    tf7 = create_content_textbox(slide7, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf7, "Why We Need Step 2 - The Template Trap [Kiela et al., 2020; Mei et al., 2024]:", 
                     "On the internet, harmless memes and hateful memes often use the exact same picture (like a popular movie scene or cute cartoon). In Step 1, the AI can still get fooled by looking only at the surface image.", is_first=True)

    add_bullet_point(tf7, "Locking the Main Brain - Safe Tuning [Cao et al., ACM MM 2023]:", 
                     "In Step 2, we completely freeze the main AI so its general knowledge is 100% safe from harm. We only tune the small translator (MLP) and the scoring judge (LRC).")
    add_sub_bullet(tf7, "This lets us focus all computational power on organizing the meme memory without any risk of damaging the AI's core brain.")

    add_bullet_point(tf7, "Learning by Comparing - Contrastive Loss [Schroff et al., 2015; Chen et al., 2020]:", 
                     "The AI looks at pairs of memes to learn what truly belongs together based on meaning using InfoNCE contrastive loss (L_CL):")
    add_sub_bullet(tf7, "It pulls pseudo-gold positive memes with the same hateful intent close together in its memory space [Mei et al., ACL 2024].")
    add_sub_bullet(tf7, "It pushes harmless memes and unrelated topics far away into separate corners.")

    add_bullet_point(tf7, "Hard Negative Mining via FAISS [Johnson et al., IEEE 2021]:", 
                     "It uses lightning-fast search (FAISS) to dig up 'trick memes'—memes that look identical on the surface but mean opposite things—teaching the AI to spot subtle traps.")


    # =========================================================
    # SLIDE 8: INFERENCE VIA RKC
    # =========================================================
    slide8 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide8, 8, 14)
    set_slide_title(slide8, "Handling New Memes: Searching Past Examples with RKC")

    tf8 = create_content_textbox(slide8, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf8, "The Challenge of Real Internet Chaos [Cao et al., WWW 2024]:", 
                     "New viral jokes and memes blow up every single day. Retraining giant multi-billion-parameter AI models every week is way too slow and costs millions of dollars.", is_first=True)

    add_bullet_point(tf8, "The 3-Step RKC Inference Flow [Khandelwal et al., ICLR 2020; Mei et al., 2025]:", 
                     "Instead of guessing blindly, the AI checks its memory bank using a quick 3-step lookup:")
    add_sub_bullet(tf8, "1. Create Fingerprint: When an unfamiliar meme appears, the AI extracts embedding g_t from the frozen model.")
    add_sub_bullet(tf8, "2. Search Past Examples: FAISS instantly searches the database to find the K closest matching memes it has seen before.")
    add_sub_bullet(tf8, "3. Similarity-Weighted Vote: The closest past memes vote on whether this new one is hateful, weighted by cosine similarity.")

    add_bullet_point(tf8, "Instant Updates in Seconds - Zero Retraining [Mei et al., EMNLP 2025]:", 
                     "When a new wave of hate memes appears during an election or breaking news event, content moderators don't need to retrain the AI.")
    add_sub_bullet(tf8, "They just add 10 to 20 verified examples into the database. In literally seconds, the AI catches on to the new trend without spending a dime on training GPUs.")


    # =========================================================
    # SLIDE 9: DATASETS
    # =========================================================
    slide9 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide9, 9, 14)
    set_slide_title(slide9, "Putting It to the Test: 6 Diverse Real-World Datasets")

    tf9 = create_content_textbox(slide9, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf9, "Testing on 6 Real-World Problem Areas:", 
                     "To prove this works in the wild and not just in a test lab, the team evaluated it across six established public meme collections:", is_first=True)

    add_sub_bullet(tf9, "1. HatefulMemes [Meta AI; Kiela et al., NeurIPS 2020]: Over 10,000 deceptive trick memes where the picture or text alone looks harmless.")
    add_sub_bullet(tf9, "2. HarMeme [Pramanick et al., EMNLP 2021]: COVID-19 memes attacking doctors, spreading health hoaxes, or blaming specific racial groups.")
    add_sub_bullet(tf9, "3. MAMI [Fersini et al., SemEval 2022]: Memes focused on misogyny, online harassment, and sexism against women.")
    add_sub_bullet(tf9, "4. Harm-P [Pramanick et al., NAACL 2021]: Toxic political memes using dirty tricks, fake stories, and personal character attacks.")
    add_sub_bullet(tf9, "5. MultiOFF [Suryawanshi et al., 2020]: A wide mix of everyday offensive memes pulled directly from social media.")
    add_sub_bullet(tf9, "6. PrideMM [Shah et al., ACM MM 2024]: Hate memes specifically attacking and bullying LGBTQ+ communities.")

    add_bullet_point(tf9, "How Performance Was Scored [Mei et al., 2025]:", 
                     "Models were tested both on memes from the same topic and on completely new meme categories they had never seen before.")
    add_sub_bullet(tf9, "Scored using standard Accuracy (percentage of correct calls) and AUC-ROC (how reliably the AI separates hate from safe content without guessing).")


    # =========================================================
    # SLIDE 10: TABLE 1 - SUPERVISED RESULTS
    # =========================================================
    slide10 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide10, 10, 14)
    set_slide_title(slide10, "Results: Setting New Records Across All Datasets")

    # Table 1 image on left half
    t1_width = Inches(4.5)
    t1_height = Inches(3.4)
    slide10.shapes.add_picture(images["table1"], Inches(0.6), Inches(1.45), width=t1_width, height=t1_height)

    # Explanatory text on right half
    tf10 = create_content_textbox(slide10, Inches(5.3), Inches(1.4), Inches(4.2), Inches(5.3))
    add_bullet_point(tf10, "Top Scores Across All 6 Datasets [Mei et al., 2025]:", 
                     "RA-HMD took the #1 spot on every single benchmark, easily beating standard training methods and models given only a few example prompts.", is_first=True, pt_heading=14.5, pt_body=13, space_after=5)

    add_bullet_point(tf10, "Crushing Models 8 Times Bigger [PaLI-X; Chen et al., 2023]:", 
                     "Our 7-billion parameter model (Qwen2-VL) reached an 87.2% score on Facebook's HatefulMemes, beating Google's giant 55-billion parameter PaLI-X (85.5%). It's much smarter while being 8x smaller!", pt_heading=14.5, pt_body=13, space_after=5)

    add_bullet_point(tf10, "Clear Leap Over Previous Best [RGCL; Mei et al., ACL 2024]:", 
                     "It beats the previous best system (RGCL) by over +4.0% in AUC and +3.2% in accuracy on the hardest benchmark.", pt_heading=14.5, pt_body=13, space_after=5)

    add_bullet_point(tf10, "Statistically Significant Gains [p < 0.05]:", 
                     "Strict statistical checks (paired t-test, p-values below 0.05) prove these performance gains are 100% genuine and not just random chance.", pt_heading=14.5, pt_body=13, space_after=3)


    # =========================================================
    # SLIDE 11: TABLE 2 - OUT-OF-DOMAIN RESULTS
    # =========================================================
    slide11 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide11, 11, 14)
    set_slide_title(slide11, "Results on New Trends: Handling Unknown Memes")

    # Table 2 image on left half
    t2_width = Inches(4.6)
    t2_height = Inches(2.7)
    slide11.shapes.add_picture(images["table2"], Inches(0.6), Inches(1.5), width=t2_width, height=t2_height)

    # Explanatory text on right half
    tf11 = create_content_textbox(slide11, Inches(5.4), Inches(1.4), Inches(4.1), Inches(5.3))
    add_bullet_point(tf11, "Testing on Completely Unseen Topics [Huang et al., 2024]:", 
                     "To see how the AI survives in the wild, it was trained on one topic (like COVID memes) and tested on totally different topics (like political or misogynistic memes) without any warning.", is_first=True, pt_heading=14.5, pt_body=13, space_after=5)

    add_bullet_point(tf11, "Memory Search Beats Simple Prompts [Hee et al., 2024]:", 
                     "Using the memory search (RKC) beat standard 4-example prompting across the board, proving that looking up real past cases works far better than just giving the AI a quick hint.", pt_heading=14.5, pt_body=13, space_after=5)

    add_bullet_point(tf11, "Beating Frontier Models [OpenAI GPT-4o, 2024]:", 
                     "When facing brand-new meme topics, RA-HMD scored higher accuracy than leading commercial models like OpenAI's GPT-4o.", pt_heading=14.5, pt_body=13, space_after=5)

    add_bullet_point(tf11, "A True Shield for Real Social Media [Mei et al., 2025]:", 
                     "This proves that having an external memory bank shields content moderation teams from getting blindsided by new viral trends.", pt_heading=14.5, pt_body=13, space_after=3)


    # =========================================================
    # SLIDE 12: KNOWLEDGE RETENTION & EXPLANATIONS (FIG 1)
    # =========================================================
    slide12 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide12, 12, 14)
    set_slide_title(slide12, "Keeping Common Sense & Writing Clear Explanations")

    # Figure 1 on the left side
    f1_width = Inches(3.9)
    f1_height = Inches(2.5)
    slide12.shapes.add_picture(images["fig1"], Inches(0.7), Inches(1.5), width=f1_width, height=f1_height)

    # Card beneath Figure 1 summarizing MMMU preservation
    card_mmmu = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.2), Inches(3.9), Inches(2.5))
    card_mmmu.fill.solid()
    card_mmmu.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card_mmmu.line.color.rgb = COLOR_BORDER
    tf_mmmu = card_mmmu.text_frame
    tf_mmmu.word_wrap = True
    tf_mmmu.margin_left = tf_mmmu.margin_top = tf_mmmu.margin_right = tf_mmmu.margin_bottom = Inches(0.12)
    p_m0 = tf_mmmu.paragraphs[0]
    p_m0.text = "General Intelligence Test (Table 4):"
    p_m0.font.bold = True
    p_m0.font.size = Pt(13)
    p_m0.font.color.rgb = COLOR_HEADING
    p_m0.space_after = Pt(3)

    p_m1 = tf_mmmu.add_paragraph()
    p_m1.text = "• Normal Untrained AI: MMMU 40.2 | GQA 60.4\n• Regular Fine-Tuned AI: MMMU 39.1 (got worse!) | GQA 57.0 (got worse!)\n• RA-HMD (Our Model): MMMU 40.4 (stayed sharp!) | GQA 60.1 (stayed sharp!)"
    p_m1.font.size = Pt(11.5)
    p_m1.font.color.rgb = COLOR_BODY
    p_m1.space_after = Pt(3)

    p_m2 = tf_mmmu.add_paragraph()
    p_m2.text = "Our method learns memes without losing its general common sense."
    p_m2.font.size = Pt(10.5)
    p_m2.font.bold = True
    p_m2.font.color.rgb = COLOR_ACCENT

    # Explanatory text on the right side
    tf12 = create_content_textbox(slide12, Inches(4.9), Inches(1.4), Inches(4.5), Inches(5.3))
    add_bullet_point(tf12, "Humans Preferred Our Explanations 61% of the Time [Hee et al., 2023]:", 
                     "In blind tests where human judges didn't know which AI wrote what, people preferred RA-HMD's explanations 61% of the time, compared to only 14% for standard training (25% were ties).", is_first=True, pt_heading=15, pt_body=13.5, space_after=6)

    add_bullet_point(tf12, "Clear, Honest Reasoning - LLM-as-Judge [Yang et al., 2023]:", 
                     "Standard AI often makes up repetitive or fake excuses. RA-HMD writes clear, step-by-step explanations explaining exactly why the combination of words and picture is offensive (scoring 5.6/10 vs 4.9/10).", pt_heading=15, pt_body=13.5, space_after=6)

    add_bullet_point(tf12, "Zero Loss in Everyday Smarts [MMMU / GQA Benchmarks]:", 
                     "It keeps its full visual problem-solving skills on standard tests (MMMU, SEEDBench, GQA), proving you don't have to break an AI just to teach it memes.", pt_heading=15, pt_body=13.5, space_after=6)

    add_bullet_point(tf12, "Essential for Fair Appeals & Moderation:", 
                     "When someone asks 'Why was my post taken down?', human review teams can read clear, sensible reasons rather than trusting an opaque black-box algorithm.", pt_heading=15, pt_body=13.5, space_after=3)


    # =========================================================
    # SLIDE 13: CONCLUSION & REAL-WORLD IMPACT
    # =========================================================
    slide13 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide13, 13, 14)
    set_slide_title(slide13, "Summary & Real-World Impact on Social Media Safety")

    tf13 = create_content_textbox(slide13, Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.3))
    add_bullet_point(tf13, "Solving the Big AI Trade-Off [Mei et al., Cambridge 2025]:", 
                     "RA-HMD proves that an AI can become a master at spotting toxic memes without losing its general everyday common sense.", is_first=True)

    add_bullet_point(tf13, "Record-Breaking Accuracy with High Efficiency:", 
                     "Sets new top scores across 6 major datasets. A compact 7-billion parameter model beat massive 55-billion parameter systems, saving huge amounts of computing power.")

    add_bullet_point(tf13, "Ready for the Fast-Moving Internet [RKC Memory Lookup]:", 
                     "By searching a database of past memes (RKC), platforms can teach the AI about new viral hate trends in seconds by simply adding examples—no costly retraining required.")

    add_bullet_point(tf13, "Transparent, Trustworthy & Fair [Human Evaluation 61%]:", 
                     "Instead of acting like an unexplainable black box, the AI writes articulate explanations that humans prefer 61% of the time, making moderation decisions fair and appealable.")

    add_bullet_point(tf13, "What's Next (Future Directions):", 
                     "Taking this approach to other languages and cultures, short-form video clips (like TikTok and Instagram Reels), and live real-time moderation systems.")


    # =========================================================
    # SLIDE 14: REFERENCES & BASE RESEARCH PAPER
    # =========================================================
    slide14 = prs.slides.add_slide(prs.slide_layouts[5])
    apply_slide_theme(slide14, 14, 14)
    set_slide_title(slide14, "References & Base Research Paper")

    # 1. Top Card: Highlighted Base Research Paper
    card_base = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(8.4), Inches(2.25))
    card_base.fill.solid()
    card_base.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card_base.line.color.rgb = COLOR_ACCENT
    card_base.line.width = Pt(1.5)

    # Accent pill on top of card
    pill_base = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.05), Inches(1.5), Inches(3.6), Inches(0.38))
    pill_base.fill.solid()
    pill_base.fill.fore_color.rgb = RGBColor(238, 246, 255)
    pill_base.line.color.rgb = RGBColor(186, 224, 255)
    p_pb = pill_base.text_frame.paragraphs[0]
    p_pb.text = "★ PRIMARY BASE RESEARCH PAPER"
    p_pb.font.size = Pt(11)
    p_pb.font.bold = True
    p_pb.font.color.rgb = COLOR_ACCENT
    p_pb.font.name = "Calibri"
    p_pb.alignment = PP_ALIGN.CENTER

    tf_base = create_content_textbox(slide14, Inches(1.05), Inches(1.95), Inches(7.9), Inches(1.55))
    p_b1 = tf_base.paragraphs[0]
    p_b1.text = "Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection"
    p_b1.font.size = Pt(15)
    p_b1.font.bold = True
    p_b1.font.color.rgb = COLOR_HEADER_BG
    p_b1.font.name = "Calibri"
    p_b1.space_after = Pt(3)

    p_b2 = tf_base.add_paragraph()
    p_b2.text = "Authors: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne"
    p_b2.font.size = Pt(12)
    p_b2.font.bold = True
    p_b2.font.color.rgb = COLOR_HEADING
    p_b2.font.name = "Calibri"
    p_b2.space_after = Pt(2)

    p_b3 = tf_base.add_paragraph()
    p_b3.text = "Affiliation & Conference: Department of Engineering, University of Cambridge  |  EMNLP 2025 (Oral Presentation)"
    p_b3.font.size = Pt(11.5)
    p_b3.font.bold = False
    p_b3.font.color.rgb = COLOR_ACCENT
    p_b3.font.name = "Calibri"
    p_b3.space_after = Pt(2)

    p_b4 = tf_base.add_paragraph()
    p_b4.text = "Preprint / Archive: arXiv:2502.13061v4 [cs.CL]  |  Official Open-Source Code: https://github.com/JingbiaoMei/RGCL"
    p_b4.font.size = Pt(10.5)
    p_b4.font.color.rgb = COLOR_MUTED
    p_b4.font.name = "Calibri"

    # 2. Left Column: Methodological & Model References
    card_ref_l = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.75), Inches(4.1), Inches(3.05))
    card_ref_l.fill.solid()
    card_ref_l.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card_ref_l.line.color.rgb = COLOR_BORDER
    card_ref_l.line.width = Pt(1.0)

    tf_rl = create_content_textbox(slide14, Inches(0.95), Inches(3.85), Inches(3.8), Inches(2.85))
    p_rl0 = tf_rl.paragraphs[0]
    p_rl0.text = "Foundational Models & Methods:"
    p_rl0.font.bold = True
    p_rl0.font.size = Pt(12.5)
    p_rl0.font.color.rgb = COLOR_HEADING
    p_rl0.font.name = "Calibri"
    p_rl0.space_after = Pt(4)

    refs_methods = [
        ("• RGCL (Precursor Framework): ", "Mei et al., 'Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning', ACL 2024."),
        ("• LoRA (Stage 1 Adaptation): ", "Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models', ICLR 2022."),
        ("• FAISS Search Engine: ", "Johnson et al., 'Billion-scale similarity search with GPUs', IEEE Trans. Big Data 2021."),
        ("• CLIP Architecture: ", "Radford et al., 'Learning Transferable Visual Models From Natural Language Supervision', ICML 2021."),
        ("• Qwen2-VL Backbone: ", "Wang et al., 'Qwen2-VL: Enhancing Vision-Language Capabilities', arXiv 2024.")
    ]
    for bold_prefix, desc in refs_methods:
        p = tf_rl.add_paragraph()
        p.space_after = Pt(3)
        p.line_spacing = 1.12
        r_b = p.add_run()
        r_b.text = bold_prefix
        r_b.font.bold = True
        r_b.font.size = Pt(10.5)
        r_b.font.color.rgb = COLOR_HEADING
        r_b.font.name = "Calibri"

        r_t = p.add_run()
        r_t.text = desc
        r_t.font.bold = False
        r_t.font.size = Pt(9.5)
        r_t.font.color.rgb = COLOR_BODY
        r_t.font.name = "Calibri"

    # 3. Right Column: Datasets & Benchmark References
    card_ref_r = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.1), Inches(3.75), Inches(4.1), Inches(3.05))
    card_ref_r.fill.solid()
    card_ref_r.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card_ref_r.line.color.rgb = COLOR_BORDER
    card_ref_r.line.width = Pt(1.0)

    tf_rr = create_content_textbox(slide14, Inches(5.25), Inches(3.85), Inches(3.8), Inches(2.85))
    p_rr0 = tf_rr.paragraphs[0]
    p_rr0.text = "Datasets & Evaluation Benchmarks:"
    p_rr0.font.bold = True
    p_rr0.font.size = Pt(12.5)
    p_rr0.font.color.rgb = COLOR_HEADING
    p_rr0.font.name = "Calibri"
    p_rr0.space_after = Pt(4)

    refs_data = [
        ("• Hateful Memes Dataset: ", "Kiela et al., 'The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes', NeurIPS 2020 (Meta AI)."),
        ("• HarMeme & Harm-P: ", "Pramanick et al., 'Detecting Harmful Memes on COVID-19 / Politics', EMNLP 2021 & NAACL 2021."),
        ("• MAMI Benchmark: ", "Fersini et al., 'SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification', SemEval 2022."),
        ("• MMMU IQ Benchmark: ", "Yue et al., 'MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark', ICLR 2024."),
        ("• Rationale Explanations: ", "Hee et al., 'Decoding the Underlying Meaning of Multimodal Hateful Memes', arXiv 2023.")
    ]
    for bold_prefix, desc in refs_data:
        p = tf_rr.add_paragraph()
        p.space_after = Pt(3)
        p.line_spacing = 1.12
        r_b = p.add_run()
        r_b.text = bold_prefix
        r_b.font.bold = True
        r_b.font.size = Pt(10.5)
        r_b.font.color.rgb = COLOR_HEADING
        r_b.font.name = "Calibri"

        r_t = p.add_run()
        r_t.text = desc
        r_t.font.bold = False
        r_t.font.size = Pt(9.5)
        r_t.font.color.rgb = COLOR_BODY
        r_t.font.name = "Calibri"

    # Save to all target file paths so any opened file is updated
    prs.save("RGCL_Paper_Presentation.pptx")
    prs.save("RGCL_Paper_Presentation_Perfect.pptx")
    prs.save("RGCL_Paper_Presentation_Final.pptx")
    print("Successfully saved updated 14-slide presentation with citations to all PPTX targets.")


if __name__ == '__main__':
    images = extract_and_crop_images("paper.pdf")
    build_presentation(images)
