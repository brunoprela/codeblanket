export const stateSpaceModelsDiscussionQuestions = [
  {
    id: 1,
    question:
      "A central bank wants to estimate the 'natural rate of unemployment' (NAIRU) which is unobserved but affects inflation. Design a state space model where: NAIRU evolves as random walk, inflation depends on unemployment gap (actual - NAIRU) and lagged inflation (Phillips curve). Address model specification, identification challenges, policy implications, and how to validate estimates.",
    answer: `[Outline: State equation: NAIRU_t = NAIRU_{t-1} + η_t; Observation: π_t = α·π_{t-1} + β·(U_t - NAIRU_t) + ε_t; Identification via Kalman filter; Challenges: NAIRU not directly observable, parameter uncertainty; Validation: check if residuals white noise, compare to survey expectations; Policy use: time-varying NAIRU improves inflation forecasting and guides monetary policy.]`,
  },
  {
    id: 2,
    question:
      "Compare structural time series models (trend + seasonal + cycle) vs ARIMA for forecasting monthly retail sales. Under what conditions would structural model be preferred? Address interpretability, forecasting performance, handling of COVID-19 structural break, and computational requirements.",
    answer: `[Answer: Structural model decomposes into interpretable components (trend up, seasonal December spike), handles level shifts better (COVID break), but more parameters to estimate; ARIMA more parsimonious, better short-term forecasts, but black-box; Structural preferred when: need decomposition, structural breaks, policy analysis; ARIMA for pure forecasting accuracy; COVID: structural model can incorporate intervention variable, ARIMA needs retraining.]`,
  },
  {
    id: 3,
    question:
      "Explain why state space models can handle missing data naturally while ARIMA cannot. Design a trading system that uses this property to combine irregularly-spaced alternative data (credit card transactions, satellite imagery) with regular daily stock prices for alpha generation.",
    answer: `[Explanation: State space uses Kalman filter which updates only when observations available; Missing data = skip update step, state propagates forward; ARIMA requires complete time series (must impute or drop observations); Trading system: Alternative data as additional observations in state vector, available irregularly; State: [alpha_t, beta_t], Observations: [return_t, alt_data_t]; Alt data improves state estimates even when sparse; Implementation challenges: observation noise calibration, data alignment, real-time vs batch.]`,
  },
];

