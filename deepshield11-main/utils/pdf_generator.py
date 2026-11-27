# Purpose: Generates forensic PDF reports for image, video, and audio deepfake/AI analysis results using FPDF2.

import os
import time
from fpdf import FPDF

class DeepShieldPDF(FPDF):
    def header(self):
        self.set_fill_color(10, 15, 29) # #0a0f1d
        self.rect(0, 0, 210, 297, 'F') # Dark mode page background
        
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(16, 185, 129) # #10b981 primary accent
        self.cell(0, 10, "DEEPSHIELD FORENSIC AUDIT REPORT", ln=True, align="C")
        
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(148, 163, 184) # #94a3b8 text color
        self.cell(0, 5, "Multi-Modal AI Generated Content Verification Platform", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f"DeepShield Cryptographic Verification System | Page {self.page_no()}", align="C")

def generate_pdf_report(record: dict, output_path: str) -> str:
    """
    Generates a dark-themed forensic audit report PDF file.
    
    Args:
        record (dict): Analysis record dict containing prediction, confidence, risk_level, media_type, factors, etc.
        output_path (str): File destination path for the PDF.
    
    Returns:
        str: Output file path.
    """
    pdf = DeepShieldPDF()
    pdf.add_page()
    
    scan_id = record.get("scan_id", "DS-000000")
    filename = record.get("filename", "Unknown_Media")
    media_type = record.get("media_type", "image").upper()
    prediction = record.get("prediction", "UNKNOWN").upper()
    confidence = record.get("confidence", 0.0)
    risk_level = record.get("risk_level", "Low").upper()
    proc_time = record.get("processing_time", 0.0)
    model_used = record.get("model_used", record.get("model", "DeepShield AI Engine"))
    timestamp = record.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC"))
    file_hash = record.get("file_hash", record.get("hash", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))

    # Section 1: Summary Box
    pdf.set_fill_color(15, 23, 42) # #0f172a card background
    pdf.rect(10, 30, 190, 45, 'F')
    
    pdf.set_xy(15, 35)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(90, 8, f"SCAN ID: {scan_id}")
    pdf.cell(90, 8, f"MEDIA TYPE: {media_type}", ln=True, align="R")
    
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(90, 6, f"Target File: {filename}")
    pdf.cell(90, 6, f"Timestamp: {timestamp}", ln=True, align="R")
    
    pdf.set_x(15)
    pdf.cell(180, 6, f"SHA-256 Hash: {file_hash[:40]}...", ln=True)

    # Section 2: Verdict Highlights
    pdf.ln(10)
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(10, 82, 190, 40, 'F')
    
    pdf.set_xy(15, 87)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(248, 250, 252)
    pdf.cell(90, 8, "DETECTION VERDICT:")
    
    # Verdict color
    if "FAKE" in prediction or "DEEPFAKE" in prediction:
        pdf.set_text_color(239, 68, 68) # #ef4444 Red
    else:
        pdf.set_text_color(16, 185, 129) # #10b981 Green
        
    pdf.cell(90, 8, f"{prediction}", ln=True, align="R")
    
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(90, 6, f"Confidence Score: {confidence:.2f}%")
    pdf.cell(90, 6, f"Threat Risk Level: {risk_level}", ln=True, align="R")
    
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(90, 6, f"Model Architecture: {model_used}")
    pdf.cell(90, 6, f"Execution Time: {proc_time:.2f} seconds", ln=True, align="R")

    # Section 3: Multimodal Detailed Breakdown (if video/audio)
    if "video_details" in record or "audio_details" in record or "factors" in record:
        pdf.ln(15)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 8, "DEEPFORENSICS FACTOR BREAKDOWN", ln=True)
        pdf.ln(2)
        
        factors = record.get("factors", {})
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(90, 6, f"Facial Alignment Anomaly: {factors.get('face', 0.0)}%")
        pdf.cell(90, 6, f"Acoustic Waveform Divergence: {factors.get('voice', 0.0)}%", ln=True)
        pdf.cell(90, 6, f"Generative Grid Artifacts: {factors.get('gan', 0.0)}%")
        pdf.cell(90, 6, f"Metadata Consistency: {factors.get('meta', 100.0)}%", ln=True)

    # Output PDF file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf.output(output_path)
    return output_path

if __name__ == "__main__":
    test_record = {
        "scan_id": "DS-TEST99",
        "filename": "sample_deepfake_video.mp4",
        "media_type": "video",
        "prediction": "Fake",
        "confidence": 96.45,
        "risk_level": "High",
        "processing_time": 1.85,
        "model_used": "VideoMAE + Wav2Vec2 Multimodal Pipeline",
        "factors": {"face": 94.2, "voice": 88.5, "gan": 91.0, "meta": 10.0}
    }
    out = generate_pdf_report(test_record, "test_report.pdf")
    print(f"Generated sample report at: {out}")
