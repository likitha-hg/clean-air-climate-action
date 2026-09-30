# Clean Air & Climate Action

**AI-powered environmental intelligence for collaborative climate action**

Clean Air & Climate Action is an AI-powered, federated climate action platform designed to combine ground-level air-quality observations, citizen-sourced environmental evidence, meteorological data, satellite fire observations, machine-learning forecasts, and regional intelligence into a unified system for pollution monitoring and coordinated response.

The platform focuses on identifying pollution signals that may be missed by macro-level monitoring alone, forecasting near-term PM2.5 conditions, detecting pollution hotspots, connecting satellite observations with ground measurements, incorporating citizen evidence, and supporting environmental authorities with actionable intelligence.

---

## The Problem

Major Indian cities monitor macro-level air quality but consistently miss hyper-local pollution events — industrial emissions, large-scale agricultural burning, seasonal smog.

The absence of real-time, granular data prevents coordinated climate action and directly threatens public health.

## The Challenge

Build an AI-powered, federated climate action platform that combines citizen-sourced data — including photos and local sensor readings — with satellite imagery and meteorological data.

The system should:

- Detect hidden pollution hotspots.
- Forecast air-quality spikes.
- Monitor environmental conditions across major economic corridors.
- Connect satellite fire observations with ground-level pollution signals.
- Incorporate citizen-submitted environmental evidence.
- Alert relevant authorities for rapid investigation and intervention.
- Enable Indian cities and states to share predictive intelligence through a federated architecture.
- Support interoperability and coordinated climate action.

---

## Platform Capabilities

### Air Quality Monitoring

The platform consolidates station-level air-quality observations and provides an interactive monitoring interface for exploring current pollution conditions.

Key pollutants include:

- PM2.5
- PM10
- NO₂

Interactive geographic visualizations provide spatial context for monitoring locations and pollution intensity.

### PM2.5 Forecasting

Machine-learning models are used to estimate PM2.5 concentrations approximately **three hours ahead**.

The forecasting workflow incorporates historical air-quality measurements and engineered temporal and environmental features to provide near-term pollution intelligence.

The project includes XGBoost-based PM2.5 forecasting models and supporting evaluation outputs.

### Pollution Hotspot Detection

The hotspot intelligence layer identifies locations showing elevated or unusual pollution signals.

Stations are assigned hotspot scores and risk levels that help prioritize areas requiring closer investigation.

This allows the platform to move beyond simple pollutant measurements toward environmental event detection.

### Satellite Fire Intelligence

Satellite fire observations provide additional spatial and temporal context for pollution events.

The platform uses **NASA FIRMS** fire observations and connects recent satellite-detected fire activity with nearby pollution signals.

Satellite-to-pollution associations are treated as **screening evidence for investigation**, not proof that a particular fire caused a pollution event.

### Citizen AI

Citizen AI incorporates local environmental observations into the monitoring workflow.

Citizen reports can include:

- Environmental photographs
- Descriptions of observed conditions
- User-provided local sensor readings
- Manually entered local measurements
- Demonstration sensor readings

**Gemini multimodal AI** analyzes citizen-submitted visual and contextual evidence and produces structured environmental observations.

Citizen evidence can then be compared with nearby monitoring-station conditions to provide additional local context.

### Authority Alert Intelligence

Environmental signals are consolidated into an authority-oriented alert layer.

Alert generation considers information such as:

- Current pollution conditions
- Hotspot scores
- Recent event evidence
- Satellite-linked context
- Environmental signal severity

Alerts are prioritized to help authorities identify locations that may require investigation or intervention.

### Federated Climate Intelligence

The platform includes a federated modeling architecture designed to support collaborative environmental intelligence without requiring all regional data to be centralized into a single training dataset.

The current regional network represents:

- North India
- South India
- East India
- West India
- Central India

Regional clients contribute model intelligence to a shared coordination layer, supporting a framework in which cities and states can collaborate while retaining regional data ownership and operational independence.

### Economic Corridor Intelligence

The Climate Action layer extends pollution intelligence beyond individual stations by organizing environmental signals around major economic corridors.

Corridor intelligence combines station membership, pollution conditions, forecasting information, and regional signals to support broader environmental coordination.

This provides a foundation for understanding pollution patterns that may span administrative boundaries and require coordinated action.

### Interoperability

The platform is designed around interoperable environmental intelligence.

Its federated architecture provides a foundation for:

- City-to-city model collaboration
- State-level environmental coordination
- Shared predictive intelligence
- Cross-region climate response
- Economic-corridor monitoring
- Broader BRICS-oriented environmental interoperability

The objective is to enable participating regions to exchange useful model intelligence and environmental signals while supporting decentralized data governance.

---

## Data Sources

The platform combines multiple environmental data streams.

### OpenAQ

**OpenAQ** provides ground-level air-quality observations used for station monitoring, data preparation, model development, and pollution analysis.

### Open-Meteo

**Open-Meteo** provides meteorological context used alongside air-quality observations.

Weather information helps provide environmental context for forecasting and pollution-event analysis.

### NASA FIRMS

**NASA FIRMS (Fire Information for Resource Management System)** provides satellite-derived fire observations used in the event-intelligence pipeline.

### Citizen-Sourced Data

Citizen observations extend the monitoring network with local evidence that may not be captured by fixed monitoring stations.

Supported evidence includes photographs, descriptions, and local sensor measurements.

---

## System Workflow

The platform follows an environmental intelligence workflow:

```text
Ground Air-Quality Data
        │
        ├──────────────┐
        │              │
        ▼              ▼
Data Cleaning      Weather Context
        │              │
        └──────┬───────┘
               ▼
        Feature Engineering
               │
               ▼
       PM2.5 Forecasting
               │
               ├───────────────┐
               ▼               ▼
      Hotspot Detection   Federated Models
               │               │
               ▼               ▼
      Event Intelligence  Regional Coordination
               │
        ┌──────┴─────────┐
        ▼                ▼
Satellite FIRMS      Citizen AI
Evidence             Evidence
        │                │
        └───────┬────────┘
                ▼
        Authority Alerts
                │
                ▼
     Climate Action Intelligence



---

## Author

**Likitha H G**

Clean Air & Climate Action

---

## Live Application

The deployed Clean Air & Climate Action platform is available on Streamlit Community Cloud:

https://clean-air-climate-action.streamlit.app/
