# Coursera Project Deliverables — SmartResolve

This folder contains the complete submission package for the **Project Summary (maximum 5 pages)** and **Screen Recording Demonstration** requirements:

## Deliverables in this Directory:

1. **`Project_Summary_SmartResolve.pdf`**:
   - The compiled PDF document (strictly **5 pages**).
   - Contains:
     * **Title of Project**: *SmartResolve: AI-Based Smart Helpdesk Ticketing System*
     * **Screenshot of tool where mentioned (Asana sprint board)**
     * **Software / Tool Used (Asana)**
     * **Objectives and Requirements (FR-1 to FR-6, NFR-1 to NFR-4)**
     * **Technical & Executive Summary (Architecture, Benchmarks, and Quantitative Results)**
     * **Screenshots of Project (Dashboard, Submission Form, Diagnostic Modal, Benchmark Studio)**
     * **Delivery & Conclusion Summary**

2. **Screen Recording Walkthroughs**:
   - **`smartresolve_demo.mp4`**: Universal H.264 MP4 video (1920x1092 Full HD, ~40 seconds) showing the complete live implementation:
     - Operations Dashboard KPIs, charts, and interactive filtering.
     - Live complaint submission with impact scope and urgency.
     - Automated AI triage modal with category prediction, confidence meter, autoencoder reconstruction error, and SLA target.
     - Ticket manager with deep diagnostic record inspector.
     - Model evaluation studio with multi-module benchmarks and confusion metrics.
     - Prototypical few-shot category testing sandbox with real-time cosine similarity.
   - **`smartresolve_demo.webp`**: Animated WebP recording.

3. **`project_summary.md`**:
   - The complete markdown source text matching the PDF document.

4. **`generate_summary_pdf.py`**:
   - Standalone Python compilation script using `reportlab` that generates `Project_Summary_SmartResolve.pdf`.
   - Run via: `python3 coursera/generate_summary_pdf.py`.

5. **`images/`**:
   - `asana_tool_screenshot.png`: Screenshot of the Asana project management board.
   - `dashboard_overview.png`: Operations dashboard overview.
   - `submit_ticket.png`: Interactive ticket ingestion form.
   - `ticket_diagnostic_modal.png`: Real-time AI diagnostic and triage inspection card.
   - `model_evaluation.png`: Model evaluation & benchmark studio.
   - `few_shot_studio.png`: Prototypical few-shot category studio.
