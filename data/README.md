#Data Directory structure

This directry contains the data pipeeline layers for the solar generation forecasting project.

##Directory Layout


## Data Files

**Note:** Actual data files are excluded from Git for privacy reasons.

### Raw Data Format
- **Location**: `data/raw/`
- **Format**: Excel files (.xlsx)
- **Naming**: `{District}_Generation_{Year}.xlsx`
- **Columns**: Date, Year, Month, Day, District, Zone, GHI, DNI, DHI, Temperature, Wind Speed, Cloud Cover, Rainfall, PV Generation

### How to Add Data

1. Place raw Excel files in `data/raw/`
2. Run the ingestion pipeline: `python main.py --steps ingest`

### Data Sources

- **Source**: Public Utilities Commission of Sri Lanka
- **Coverage**: 24 districts in Sri Lanka
- **Time Period**: 2018-2024
- **Temporal Resolution**: 5-minute intervals (8:00 AM - 5:00 PM)


