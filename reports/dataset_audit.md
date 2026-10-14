# Forensic Dataset Audit Report: CICAndMal2017

**Date:** 2026-10-13T16:58:37.309656  
**Total CSV Files:** 2,126  
**Total Dataset Size:** 1.221 GB (1,311,278,522 bytes)  
**Total Network Flows:** 2,616,579  

## 1. Directory and Category Summary

| Category | Files | Total Flows | Share (%) | Distinct Families |
| :--- | :--- | :--- | :--- | :--- |
| **SMSmalware** | 109 | 237,133 | 9.06% | 11 |
| **Scareware** | 112 | 400,841 | 15.32% | 11 |
| **Ransomware** | 101 | 348,943 | 13.34% | 10 |
| **Benign** | 1,700 | 1,205,515 | 46.07% | 1 |
| **Adware** | 104 | 424,147 | 16.21% | 10 |

## 2. Schema and Feature Overview

- **Total Header Columns:** 85
- **Duplicate Columns Detected:** Fwd Header Length
  - `Fwd Header Length` appears at column index 40 and 61.
- **Extractor Origin:** Output of `CICFlowMeter` (ISCXFlowMeter v1/v2).

## 3. Label Breakdown and Class Distribution

| Label String | Count | Percentage (%) | True Category Classification |
| :--- | :--- | :--- | :--- |
| `BENIGN` | 1,210,210 | 46.25% | Benign |
| `ADWARE_GOOLIGAN` | 93,772 | 3.58% | Malware-Associated |
| `ADWARE_FEIWO` | 56,632 | 2.16% | Malware-Associated |
| `SCAREWARE_ANDROIDDEFENDER` | 56,440 | 2.16% | Malware-Associated |
| `RANSOMWARE_SVPENG` | 54,161 | 2.07% | Malware-Associated |
| `RANSOMWARE_PORNDROID` | 46,082 | 1.76% | Malware-Associated |
| `SCAREWARE_FAKEAPPAL` | 44,563 | 1.70% | Malware-Associated |
| `RANSOMWARE_KOLER` | 44,555 | 1.70% | Malware-Associated |
| `SMSMALWARE_NANDROBOX` | 44,517 | 1.70% | Malware-Associated |
| `ADWARE_EWIND` | 43,374 | 1.66% | Malware-Associated |
| `SCAREWARE_AVFORANDROID` | 42,448 | 1.62% | Malware-Associated |
| `SCAREWARE_AVPASS` | 40,776 | 1.56% | Malware-Associated |
| `SCAREWARE_FAKEAV` | 40,089 | 1.53% | Malware-Associated |
| `RANSOMWARE_RANSOMBO` | 39,859 | 1.52% | Malware-Associated |
| `SMSMALWARE_PLANKTON` | 39,765 | 1.52% | Malware-Associated |
| `ADWARE_DOWGIN` | 39,682 | 1.52% | Malware-Associated |
| `RANSOMWARE_CHARGER` | 39,551 | 1.51% | Malware-Associated |
| `ADWARE_SHUANET` | 39,271 | 1.50% | Malware-Associated |
| `ADWARE_KEMOGE` | 38,771 | 1.48% | Malware-Associated |
| `RANSOMWARE_SIMPLOCKER` | 36,340 | 1.39% | Malware-Associated |
| `ADWARE_YOUMI` | 36,035 | 1.38% | Malware-Associated |
| `SCAREWARE_FAKEAPP` | 34,676 | 1.33% | Malware-Associated |
| `SMSMALWARE_BIIGE` | 33,678 | 1.29% | Malware-Associated |
| `SMSMALWARE_SMSSNIFFER` | 33,618 | 1.28% | Malware-Associated |
| `SCAREWARE_FAKETAOBAO` | 33,299 | 1.27% | Malware-Associated |
| `RANSOMWARE_WANNALOCKER` | 32,701 | 1.25% | Malware-Associated |
| `ADWARE_KOODOUS` | 32,547 | 1.24% | Malware-Associated |
| `ADWARE_MOBIDASH` | 31,034 | 1.19% | Malware-Associated |
| `SCAREWARE_FAKEJOBOFFER` | 30,683 | 1.17% | Malware-Associated |
| `RANSOMWARE_JISUT` | 25,672 | 0.98% | Malware-Associated |
| `SCAREWARE_ANDROIDSPY` | 25,414 | 0.97% | Malware-Associated |
| `RANSOMWARE_LOCKERPIN` | 25,307 | 0.97% | Malware-Associated |
| `SCAREWARE_VIRUSSHIELD` | 23,716 | 0.91% | Malware-Associated |
| `SMSMALWARE_FAKENOTIFY` | 22,197 | 0.85% | Malware-Associated |
| `SCAREWARE_PENETHO` | 21,631 | 0.83% | Malware-Associated |
| `SMSMALWARE_FAKEINST` | 15,026 | 0.57% | Malware-Associated |
| `ADWARE_SELFMITE` | 13,029 | 0.50% | Malware-Associated |
| `SMSMALWARE_BEANBOT` | 12,371 | 0.47% | Malware-Associated |
| `SMSMALWARE_ZSONE` | 9,644 | 0.37% | Malware-Associated |
| `SCAREWARE` | 7,430 | 0.28% | Malware-Associated |
| `SMSMALWARE_FAKEMART` | 6,401 | 0.24% | Malware-Associated |
| `SMSMALWARE_MAZARBOT` | 6,065 | 0.23% | Malware-Associated |
| `SMSMALWARE_JIFAKE` | 5,993 | 0.23% | Malware-Associated |
| `RANSOMWARE_PLETOR` | 4,715 | 0.18% | Malware-Associated |
| `MALWARE` | 2,828 | 0.11% | Malware-Associated |
| `[BLANK]` | 5 | 0.00% | Malware-Associated |

## 4. Discrepancies Between Directory Names and Internal CSV Labels

1. **Benign Traffic within Malware Directories:**
   - In `SMSmalware-CSVs`, **419 flows** are labeled `BENIGN`. These represent system background traffic (Google Services, NTP, DNS) captured during malware execution.
   - In `Scareware-CSVs`, **4,276 flows** are labeled `BENIGN`. These represent system background traffic (Google Services, NTP, DNS) captured during malware execution.
2. **Ambiguous / Generic Labels:**
   - `MALWARE` appears 2,828 times without family specification in Scareware captures.
   - `5` rows have completely empty label values due to trailing delimiter truncations.

