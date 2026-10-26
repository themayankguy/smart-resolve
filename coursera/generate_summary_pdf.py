import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
import pymupdf

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(36, 762, "SmartResolve: AI-Based Smart Helpdesk Ticketing System")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(576, 762, "Project Summary & Agile Delivery Report (Asana)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, 756, 576, 756)
            
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 36, 576, 36)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 25, "Software/Tool Used: Asana | Coursera Project Summary")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_str)
        
        self.restoreState()

def build_pdf(filename="coursera/Project_Summary_SmartResolve.pdf"):
    # Margins: 36 pt (0.5 inch). Printable width: 612 - 72 = 540 pt. Printable height: 792 - 72 = 720 pt.
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=38,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=7
    )

    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=5,
        spaceAfter=3
    )

    h2_style = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=4,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.3,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=3.5
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.3,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=10,
        spaceAfter=2
    )

    caption_style = ParagraphStyle(
        'ImageCaption',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#475569"),
        alignment=1, # Centered
        spaceBefore=3,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#FFFFFF")
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.6,
        leading=9.6,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.6,
        leading=9.6,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE OF PROJECT, SOFTWARE/TOOL USED (ASANA) & SCREENSHOT
    # =========================================================================
    story.append(Paragraph("SmartResolve: AI-Based Smart Helpdesk Ticketing System", title_style))
    story.append(Paragraph("<b>Project Summary &amp; Agile Engineering Report</b> &nbsp;|&nbsp; Coursera Guided Project Submission", subtitle_style))
    
    # Metadata Table
    meta_data = [
        [
            Paragraph("<b>Title of Project:</b> SmartResolve (AI-Based Helpdesk Ticketing)", table_cell_style),
            Paragraph("<b>Software / Tool Used:</b> <b>Asana</b> (Agile Kanban &amp; Sprint Management)", table_cell_style),
        ],
        [
            Paragraph("<b>Authors:</b> Mayank Sawant &amp; Sayali Rajput", table_cell_style),
            Paragraph("<b>Institution:</b> AI &amp; Data Science, MPSTME, SVKM's NMIMS", table_cell_style)
        ],
        [
            Paragraph("<b>Project Domain:</b> Enterprise IT Service Management (ITSM)", table_cell_style),
            Paragraph("<b>Code Repository:</b> github.com/themayankguy/smart-resolve", table_cell_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Software / Tool Used: Asana (Project &amp; Sprint Tracking)", h1_style))
    story.append(Paragraph(
        "<b>Asana</b> was utilized as the primary software project management and collaboration tool throughout the entire lifecycle of the <b>SmartResolve</b> project. Asana enabled the development team to apply Agile/Scrum principles to artificial intelligence engineering, breaking down the complex natural language processing pipeline, deep learning anomaly detector, backend RESTful microservices, and interactive frontend into discrete user stories and deliverables.",
        body_style
    ))
    story.append(Paragraph(
        "The project workspace in Asana was organized around a structured Kanban board with five core stages: <b>Backlog</b> (conceptual items and incoming requirements), <b>Ready</b> (refined tasks scheduled for implementation), <b>In Progress</b> (active feature engineering and model training), <b>Review &amp; Validation</b> (code review, benchmark verification, and test execution), and <b>Done</b> (formally verified and merged features).",
        body_style
    ))
    
    # Asana Screenshot
    asana_img_path = "coursera/images/asana_tool_screenshot.png"
    if os.path.exists(asana_img_path):
        story.append(Image(asana_img_path, width=540, height=270))
        story.append(Paragraph(
            "<b>Figure 1: Screenshot of Tool Mentioned (Asana):</b> Sprint management board for SmartResolve illustrating live task distribution across Backlog, In Progress, Ready, Review &amp; Validation, and Done columns.",
            caption_style
        ))
    
    story.append(Paragraph(
        "<b>Agile Sprint Milestones Managed via Asana:</b> Completed 8 major deliverables including SQLite database schema setup, FastAPI routing implementation, Sentence-BERT embedding vectorization, PyTorch Autoencoder training, and full-stack smoke testing with 100% test pass rate.",
        body_style
    ))

    # Force PageBreak to strictly guarantee Page 1 boundaries
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: OBJECTIVES AND REQUIREMENTS
    # =========================================================================
    story.append(Paragraph("2. Objectives of the Project", h1_style))
    story.append(Paragraph(
        "Enterprise IT helpdesks routinely encounter high volumes of incoming support requests. Manual triage by human dispatchers leads to severe operational bottlenecks, high ticket turnaround times, routing errors, and Service Level Agreement (SLA) violations. SmartResolve automates the complaint intake, categorization, risk assessment, and escalation workflow. The core project objectives are:",
        body_style
    ))
    story.append(Paragraph("• <b>Automated Complaint Ingestion &amp; Semantic Encoding:</b> Ingest diverse natural language complaints and encode them into continuous 384-dimensional semantic representations using Sentence-BERT (<i>all-MiniLM-L6-v2</i>).", bullet_style))
    story.append(Paragraph("• <b>Multi-Class IT Domain Classification:</b> Accurately categorize tickets into 8 critical enterprise categories: Hardware, Software, Network/VPN, Account/IAM Access, Security Incident, Database/Storage, Printer and Peripheral, and Other IT Services.", bullet_style))
    story.append(Paragraph("• <b>Unsupervised Out-of-Distribution (OOD) Anomaly Detection:</b> Identify non-technical tickets, spam, out-of-scope requests, or anomalous outages without requiring labeled anomaly data, using a PyTorch Autoencoder reconstruction loss threshold (τ = 0.0018).", bullet_style))
    story.append(Paragraph("• <b>Prototypical Few-Shot Metric Learning:</b> Dynamically handle emerging edge cases and novel IT issue types from only k = 1 to 5 labeled exemplars via centroid prototype matching in cosine metric space.", bullet_style))
    story.append(Paragraph("• <b>SLA &amp; Escalation Risk Scoring:</b> Automate SLA target allocation (2h to 48h) and escalation risk scoring using a composite function of predicted category, model confidence, user priority, and organizational impact scope.", bullet_style))
    story.append(Paragraph("• <b>Ultra-Low Latency Inference:</b> Maintain sub-10 ms per-ticket inference latency on standard CPU hardware to support high-throughput enterprise ticket ingestion.", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Requirements Specification", h1_style))
    story.append(Paragraph("All technical requirements were defined, tracked, and validated using Asana tasks:", body_style))

    # Functional Requirements Table
    fr_data = [
        [Paragraph("Req ID", table_header_style), Paragraph("Requirement Category", table_header_style), Paragraph("Description &amp; Engineering Implementation", table_header_style)],
        [
            Paragraph("<b>FR-1</b>", table_cell_bold),
            Paragraph("Natural Language Intake", table_cell_style),
            Paragraph("Accept unstructured text via REST endpoint or web UI; normalize whitespace and encode via Sentence-BERT into 384-dim dense vectors.", table_cell_style)
        ],
        [
            Paragraph("<b>FR-2</b>", table_cell_bold),
            Paragraph("Multi-Class Routing", table_cell_style),
            Paragraph("Generate posterior probability distributions across 8 enterprise IT domains using regularized multinomial logistic regression.", table_cell_style)
        ],
        [
            Paragraph("<b>FR-3</b>", table_cell_bold),
            Paragraph("Anomaly Detection", table_cell_style),
            Paragraph("Pass embeddings through a 5-layer PyTorch Autoencoder (384 → 128 → 64 → 128 → 384); flag reconstruction MSE &gt; 0.0018 as anomalous.", table_cell_style)
        ],
        [
            Paragraph("<b>FR-4</b>", table_cell_bold),
            Paragraph("Few-Shot Adaptation", table_cell_style),
            Paragraph("Support rapid creation of ad-hoc support classes by calculating class prototype centroids in cosine metric space from k sample tickets.", table_cell_style)
        ],
        [
            Paragraph("<b>FR-5</b>", table_cell_bold),
            Paragraph("SLA &amp; Priority Engine", table_cell_style),
            Paragraph("Synthesize ticket priority (P1–P4) and organizational impact (Organization, Department, Individual) into deterministic SLA targets (2h–48h).", table_cell_style)
        ],
        [
            Paragraph("<b>FR-6</b>", table_cell_bold),
            Paragraph("Full-Stack Dashboard", table_cell_style),
            Paragraph("Provide IT agents with real-time analytics, queue management, model evaluation visualizations, and ticket status resolution toggles.", table_cell_style)
        ],
    ]
    fr_table = Table(fr_data, colWidths=[40, 110, 390])
    fr_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(fr_table)

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Non-Functional Requirements (NFRs):</b>", h2_style))
    story.append(Paragraph("• <b>NFR-1 (Latency):</b> Average end-to-end processing latency under 10 ms per ticket on commodity multi-core CPU architecture.", bullet_style))
    story.append(Paragraph("• <b>NFR-2 (Data Isolation):</b> Strict 85/15 train-test split isolation with zero data leakage between training and evaluation partitions.", bullet_style))
    story.append(Paragraph("• <b>NFR-3 (Lightweight Footprint):</b> Zero microservice sprawl; single-process asynchronous FastAPI backend with ACID SQLite persistence.", bullet_style))
    story.append(Paragraph("• <b>NFR-4 (Verification &amp; Reliability):</b> 100% automated test pass rate for backend API endpoints, database operations, and ML routines.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: TECHNICAL & EXECUTIVE SUMMARY, QUANTITATIVE RESULTS
    # =========================================================================
    story.append(Paragraph("4. Technical Summary &amp; Implementation Architecture", h1_style))
    story.append(Paragraph(
        "<b>SmartResolve</b> implements a dual-inference machine learning architecture integrated into an asynchronous enterprise web service. The technical stack comprises three unified tiers:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Machine Learning Engine:</b> Incoming complaints are transformed by a frozen Sentence-BERT transformer (<i>all-MiniLM-L6-v2</i>) into 384-dimensional embeddings. The dense vector is concurrently evaluated by: (a) an L2-regularized multinomial classifier for primary domain prediction, (b) a deep symmetric PyTorch Autoencoder (384 → 128 → 64 → 128 → 384) trained exclusively on in-distribution tickets using Mean Squared Error reconstruction loss, and (c) a Prototypical Metric Engine that computes cosine similarity against dynamic cluster centroids.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Asynchronous Backend:</b> Implemented in Python using FastAPI and SQLite, providing RESTful endpoints for real-time ticket ingestion, batch evaluation, few-shot prototype registration, and database aggregation under full ACID guarantees.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Reactive Frontend:</b> Engineered with React 19, Vite, and custom CSS, featuring live ticket distribution widgets, submission modal, diagnostic explainability cards, and interactive few-shot experimentation studios.",
        body_style
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("5. Quantitative Experimental Results &amp; Findings", h1_style))
    story.append(Paragraph(
        "Evaluation was conducted on a curated benchmark of <b>900 real customer support tickets</b> (800 in-distribution across 8 balanced categories + 100 out-of-distribution anomaly inquiries) using a strictly held-out test split (N = 135).",
        body_style
    ))

    # Metrics Table
    results_data = [
        [Paragraph("Evaluation Metric", table_header_style), Paragraph("TF-IDF Baseline", table_header_style), Paragraph("SmartResolve (Ours)", table_header_style), Paragraph("Performance Gain / Status", table_header_style)],
        [
            Paragraph("<b>Overall Test Accuracy</b>", table_cell_bold),
            Paragraph("64.44%", table_cell_style),
            Paragraph("<b>69.63%</b>", table_cell_bold),
            Paragraph("<b>+5.19% absolute gain</b> (5.5× random baseline)", table_cell_style)
        ],
        [
            Paragraph("<b>Macro F1-Score</b>", table_cell_bold),
            Paragraph("63.12%", table_cell_style),
            Paragraph("<b>69.63%</b>", table_cell_bold),
            Paragraph("Balanced performance across all 8 IT categories", table_cell_style)
        ],
        [
            Paragraph("<b>Security Incident F1</b>", table_cell_bold),
            Paragraph("74.20%", table_cell_style),
            Paragraph("<b>87.50%</b> (93.3% recall)", table_cell_bold),
            Paragraph("High-precision routing for high-impact tickets", table_cell_style)
        ],
        [
            Paragraph("<b>Account / IAM Access F1</b>", table_cell_bold),
            Paragraph("78.10%", table_cell_style),
            Paragraph("<b>86.67%</b> (86.7% recall)", table_cell_bold),
            Paragraph("High fidelity on password &amp; permission requests", table_cell_style)
        ],
        [
            Paragraph("<b>Anomaly Detection ROC-AUC</b>", table_cell_bold),
            Paragraph("52.30% (Random)", table_cell_style),
            Paragraph("<b>76.21%</b> (PR-AUC: 50.22%)", table_cell_bold),
            Paragraph("<b>8.26% False Positive Rate</b> at τ = 0.0018", table_cell_style)
        ],
        [
            Paragraph("<b>Few-Shot Accuracy (k=5)</b>", table_cell_bold),
            Paragraph("12.50% (Random)", table_cell_style),
            Paragraph("<b>44.81%</b> (3.6× baseline)", table_cell_bold),
            Paragraph("Rapid adaptation to emerging IT categories", table_cell_style)
        ],
        [
            Paragraph("<b>Mean Inference Latency</b>", table_cell_bold),
            Paragraph("14.20 ms", table_cell_style),
            Paragraph("<b>5.93 ms</b> (p95: 7.83 ms)", table_cell_bold),
            Paragraph("<b>2.4× faster</b>; exceeds sub-10ms real-time target", table_cell_style)
        ],
    ]
    res_table = Table(results_data, colWidths=[130, 85, 125, 200])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")])
    ]))
    story.append(res_table)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Summary of Findings:</b> Experimental results demonstrate that dense semantic embeddings consistently outperform traditional n-gram models on messy, real-world ticket phrasing. The unsupervised PyTorch Autoencoder achieves an operational ROC-AUC of 76.21% with an 8.26% false positive rate at τ = 0.0018, isolating non-technical noise without manual supervision.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: SCREENSHOTS OF PROJECT (PART I)
    # =========================================================================
    story.append(Paragraph("6. Screenshots of Project (Part I: Dashboard &amp; Ticket Intake)", h1_style))
    story.append(Paragraph(
        "The SmartResolve user interface provides complete operational visibility for IT technicians, queue managers, and end-users.",
        body_style
    ))

    # Dashboard Screenshot
    dash_img_path = "coursera/images/dashboard_overview.png"
    if os.path.exists(dash_img_path):
        story.append(Image(dash_img_path, width=540, height=245))
        story.append(Paragraph(
            "<b>Figure 2: SmartResolve Live Operations Dashboard:</b> Visualizing real-time ticket volume metrics (Total, Open, Escalated, Normal/Anomaly count), category distribution bar charts, priority breakdown, and the dynamic ticket queue table with live status filtering.",
            caption_style
        ))

    story.append(Spacer(1, 4))
    
    # Ticket Submission Screenshot
    submit_img_path = "coursera/images/submit_ticket.png"
    if os.path.exists(submit_img_path):
        story.append(Image(submit_img_path, width=540, height=245))
        story.append(Paragraph(
            "<b>Figure 3: Interactive Ticket Ingestion Form:</b> End-user complaint intake portal featuring natural language input, suggested category chips, priority selection, and organizational impact scope tagging (Organization, Department, Individual) for automated SLA scheduling.",
            caption_style
        ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: SCREENSHOTS OF PROJECT (PART II: AI DIAGNOSTICS & BENCHMARK) & CONCLUSION
    # =========================================================================
    story.append(Paragraph("7. Screenshots of Project (Part II: AI Diagnostics &amp; Model Benchmark)", h1_style))

    # Triage Diagnostic Modal Screenshot
    modal_img_path = "coursera/images/ticket_diagnostic_modal.png"
    if os.path.exists(modal_img_path):
        story.append(Image(modal_img_path, width=500, height=220))
        story.append(Paragraph(
            "<b>Figure 4: Automated AI Diagnostic Record &amp; Triage Modal:</b> Deep inspection card showing predicted category (Network and Wi-Fi, 77.5% confidence), reconstruction error (0.003 vs threshold 0.0047), assigned support group, and prioritization rationale.",
            caption_style
        ))

    story.append(Spacer(1, 4))

    # Model Evaluation Screenshot
    eval_img_path = "coursera/images/model_evaluation.png"
    if os.path.exists(eval_img_path):
        story.append(Image(eval_img_path, width=540, height=220))
        story.append(Paragraph(
            "<b>Figure 5: Live Model Evaluation &amp; Benchmark Studio:</b> Real-time performance dashboard displaying confusion matrix, per-category precision/recall/F1 metrics, autoencoder ROC curve, and hardware latency percentiles (5.93 ms mean, 5.32 ms median).",
            caption_style
        ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("8. Project Conclusion &amp; Deliverables Summary", h1_style))
    story.append(Paragraph(
        "<b>Delivery &amp; Compliance Summary:</b> SmartResolve was delivered successfully following Agile sprints coordinated via Asana. The project achieved 100% test passing status, sub-10ms inference speed, robust 76.21% OOD anomaly gating, and a complete open-source repository at <b>https://github.com/themayankguy/smart-resolve.git</b>.",
        body_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
