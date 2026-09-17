"""
Automation Skill.

The one real "does something, not just describes something" action in
this project: generate the business report AND save it to disk
(outputs/business_report.md + .html) - a genuine side effect, not a
simulated one. If SMTP credentials are configured in the environment
and an `email_to` address is given, it also actually emails the HTML
report via smtplib.

Nothing here ever reports success it didn't verify: the save is only
reported once the files are confirmed on disk (via generate_business_
report's own success path), and the email block only says "sent" if
smtplib's send_message call completed without raising.
"""

import os
import smtplib
from email.message import EmailMessage

from app.config import settings
from app.tools.report_tools import generate_business_report


def deliver_business_report(
    file_path: str,
    date_column: str = None,
    value_column: str = None,
    category_column: str = None,
    target_column: str = None,
    email_to: str = None
) -> dict:
    """
    Generate a business report and save it to disk (outputs/). If
    `email_to` is given and SMTP is configured (SMTP_HOST, SMTP_PORT,
    SMTP_USER, SMTP_PASSWORD, SMTP_FROM in the environment), also
    email the HTML report to that address.
    """

    report = generate_business_report(
        file_path,
        date_column=date_column,
        value_column=value_column,
        category_column=category_column,
        target_column=target_column
    )

    if "error" in report:
        return report

    saved_files = [
        report["report_path_markdown"],
        report["report_path_html"],
        report["report_path_pdf"],
        report["report_path_docx"]
    ]
    files_confirmed = all(os.path.exists(path) for path in saved_files)

    result = dict(report)
    result["action"] = "report_saved" if files_confirmed else "save_failed"
    result["saved_files"] = saved_files if files_confirmed else []

    if not files_confirmed:
        result["error"] = (
            "Report generation reported success but the output "
            "files were not found on disk."
        )
        return result

    if not email_to:
        result["email"] = {
            "attempted": False,
            "sent": False,
            "reason": "No recipient email address was provided."
        }
        return result

    if not settings.smtp_configured:
        result["email"] = {
            "attempted": False,
            "sent": False,
            "reason": (
                "SMTP is not configured. Set SMTP_HOST, SMTP_PORT, "
                "SMTP_USER, SMTP_PASSWORD (and optionally SMTP_FROM) "
                "in the environment to enable email delivery."
            )
        }
        return result

    try:
        with open(report["report_path_html"], "r", encoding="utf-8") as file:
            html_body = file.read()

        message = EmailMessage()
        message["Subject"] = "Automated Business Analysis Report"
        message["From"] = settings.smtp_from
        message["To"] = email_to
        message.set_content(
            "Your automated business report is attached as HTML. "
            "Open this email in an HTML-capable client to view it "
            "inline, or see the plain-text summary below.\n\n"
            + "\n".join(f"- {insight}" for insight in report["key_insights"])
        )
        message.add_alternative(html_body, subtype="html")

        with open(report["report_path_pdf"], "rb") as pdf_file:
            message.add_attachment(
                pdf_file.read(),
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(report["report_path_pdf"])
            )

        with smtplib.SMTP(
            settings.smtp_host, settings.smtp_port, timeout=15
        ) as server:
            if settings.smtp_use_tls:
                server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)

        result["email"] = {
            "attempted": True,
            "sent": True,
            "to": email_to
        }

    except Exception as exc:
        result["email"] = {
            "attempted": True,
            "sent": False,
            "error": str(exc)
        }

    return result