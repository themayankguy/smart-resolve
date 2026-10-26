# SmartResolve Dataset & Data Schema

This directory contains the dataset guidelines and seed data for **SmartResolve** (SIH25195: Smart Helpdesk Ticketing Solution for IT Services).

## Expected CSV Format

When uploading a dataset via the Admin UI (`POST /api/dataset/upload`), the CSV file must contain the following columns:

| Column Name | Required | Type | Description |
|---|---|---|---|
| `complaint_text` | **Yes** | string | The full text of the user's IT issue / complaint. |
| `category` | **Yes** | string | The ground-truth category label (e.g., `Network and Wi-Fi`, `Hardware`, `Software and Application`, `Account and Authentication`, `Email and Communication`, `Printer and Peripheral`, `Security Incident`, `Other / Unknown`). |
| `is_normal` | No | int (0 or 1) | `1` indicates standard IT complaints used to train normal Autoencoder representation; `0` represents anomalous, out-of-domain, or unusual complaints. Default is `1`. |
| `priority` | No | string | `Low`, `Medium`, `High`, or `Critical`. |
| `impact` | No | string | `Individual`, `Department`, or `Enterprise`. |

## Seed Dataset (`seed_complaints.csv`)

`seed_complaints.csv` is provided out of the box so the application functions immediately without requiring external uploads. It contains 120+ realistic enterprise IT complaints curated for SIH25195, balanced across categories, with normal operational examples and out-of-distribution anomaly test cases.
