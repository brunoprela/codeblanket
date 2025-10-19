/**
 * Quiz questions for Advanced Visualization Techniques section
 */

export const advancedvisualizationQuiz = [
  {
    id: 'q1',
    question:
      'Compare static visualizations (matplotlib/seaborn) with interactive visualizations (Plotly/Bokeh). When would you choose each, and what are the trade-offs?',
    hint: 'Think about audience, use case, performance, and deployment context.',
    sampleAnswer:
      "Static vs interactive visualizations serve different purposes with distinct trade-offs: STATIC VISUALIZATIONS (Matplotlib/Seaborn): WHEN TO USE: (1) Publication and reports: Papers, PDFs, presentations require static images. (2) Reproducibility: Exact same image every time, version-controlled easily. (3) Performance: Fast rendering, no browser required. (4) Batch generation: Create hundreds of plots programmatically. (5) Print media: Physical reports, posters. PROS: Simple, fast, mature libraries, extensive customization, works offline, small file sizes (PNG/PDF), consistent rendering. CONS: No interactivity (can't zoom, hover for details, toggle series), fixed view decided by analyst. INTERACTIVE VISUALIZATIONS (Plotly/Bokeh): WHEN TO USE: (1) Exploratory analysis: Let stakeholders explore data themselves. (2) Dashboards: Real-time monitoring, business intelligence. (3) Web applications: Embedded in web pages. (4) Large datasets: Zoom into regions of interest. (5) High-dimensional data: Interactive filtering and selection. PROS: Hover for details, zoom/pan, toggle series on/off, cross-filtering, animations, responsive to user input, better engagement. CONS: Requires browser/HTML, larger file sizes, can be slow with huge datasets, harder to version control (HTML files), inconsistent rendering across browsers. BEST PRACTICES: (1) Start with static for quick EDA, switch to interactive for presentation. (2) Use static for reports you'll print or publish. (3) Use interactive for dashboards and web apps. (4) For presentations, interactive can be impressive but have static backup. (5) Consider audience: Technical team may prefer static (reproduce analysis), business stakeholders prefer interactive (explore themselves). HYBRID APPROACH: Generate static plots during analysis, create interactive dashboard for final presentation. Use static in documentation, interactive in deployed applications. PERFORMANCE: Static handles millions of points easily. Interactive struggles above ~100K points without aggregation (hexbin, datashader). REAL EXAMPLE: Financial report: Static plots (reproducible, printable). Trading dashboard: Interactive (real-time, drill-down into specific securities).",
    keyPoints: [
      'Static: publications, reproducibility, performance, print media',
      'Interactive: dashboards, exploration, web apps, engagement',
      'Static pros: fast, version-controlled, offline, consistent',
      'Interactive pros: hover details, zoom, user-driven exploration',
      'Choose based on audience and use case (analysis vs presentation)',
      'Hybrid approach: static for docs, interactive for deployed apps',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the purpose of geographic visualizations in data analysis. How can spatial patterns inform feature engineering and model development?',
    sampleAnswer:
      'Geographic visualizations reveal spatial patterns that inform critical modeling decisions: PURPOSE OF GEOGRAPHIC VISUALIZATION: (1) SPATIAL CLUSTERING: Identify geographic clusters (high-value neighborhoods, disease hotspots, customer density). Example: Housing prices cluster in coastal areas. (2) PROXIMITY EFFECTS: Discover importance of location relative to landmarks (distance to city center, nearest competitor, ocean proximity). (3) REGIONAL DIFFERENCES: Reveal distinct regional behaviors requiring separate models or region-specific features. (4) OUTLIER DETECTION: Geographic outliers may indicate errors or special cases. (5) SPATIAL AUTOCORRELATION: Nearby locations have similar values - violates independence assumption. FEATURE ENGINEERING FROM SPATIAL PATTERNS: (1) DISTANCE FEATURES: If visualization shows values decrease with distance from center, create "distance_to_downtown" feature. Example: House prices highest near coast → create "distance_to_coast". (2) GEOGRAPHIC CLUSTERING: Create neighborhood/region identifiers as categorical features. Use clustering algorithms (KMeans, DBSCAN) on lat/lon to create location clusters. (3) SPATIAL DENSITY: Create "density" features (population density, competitor density within radius). (4) RELATIVE POSITION: North/South/East/West indicators, urban vs suburban vs rural. (5) SPATIAL LAGS: Include average value of neighbors (common in geospatial modeling). MODELING IMPLICATIONS: (1) SPATIAL FEATURES BECOME TOP PREDICTORS: In housing data, location often more important than house characteristics. (2) REGIONAL MODELS: If strong regional differences, train separate models per region. (3) SPATIAL CROSS-VALIDATION: Can\'t use random splits - use spatial blocks to avoid leakage. (4) GEOSPATIAL MODELS: Consider specialized models (geographically weighted regression, spatial autoregressive models). REAL EXAMPLE: Visualizing Uber ride data geographically revealed: (1) High demand clusters → create "in_high_demand_zone" feature. (2) Airport proximity matters → create "distance_to_airport". (3) Downtown vs suburbs different → create "is_downtown" feature. (4) Time-of-day patterns vary by location → interaction features. Result: 15% accuracy improvement by adding spatial features discovered through geographic visualization.',
    keyPoints: [
      'Geographic visualizations reveal spatial clusters and proximity effects',
      'Enable creation of distance-based features (to landmarks, city center)',
      'Identify regional differences requiring separate models or features',
      'Support geographic clustering for categorical location features',
      'Inform spatial cross-validation strategies to avoid leakage',
      'Spatial patterns often among strongest predictors',
    ],
  },
  {
    id: 'q3',
    question:
      'Discuss best practices for creating effective data visualizations that communicate insights to both technical and non-technical audiences. What makes a visualization publication-quality?',
    sampleAnswer:
      "Effective visualizations require both technical correctness and communication clarity: UNIVERSAL BEST PRACTICES: (1) CLEAR PURPOSE: Every plot should answer a specific question. No chart for chart's sake. (2) APPROPRIATE CHART TYPE: Bar charts for categories, scatter for relationships, line for time series, histograms for distributions. Don't use 3D pie charts! (3) CLEAN DESIGN: Remove chart junk (unnecessary grid lines, borders, 3D effects). High data-ink ratio (maximize information per ink). (4) READABLE TEXT: Large enough fonts (12+ for labels), clear axis titles, descriptive title. (5) ACCESSIBLE COLORS: Colorblind-friendly palettes (avoid red-green), sufficient contrast. (6) PROPER SCALES: Start Y-axis at zero for bar charts (unless showing changes), use appropriate log scales when needed. (7) CONTEXT: Add reference lines (averages, targets), annotate important points. (8) HONEST REPRESENTATION: Don't manipulate scales to mislead, show uncertainty when relevant. FOR NON-TECHNICAL AUDIENCES: (1) SIMPLIFIED: Focus on key insight, remove technical jargon. (2) ANNOTATED: Add text explaining what to look at. (3) STORYTELLING: Guide viewer's attention with titles and highlights. (4) FAMILIAR FORMATS: Stick to common chart types they understand. (5) ACTIONABLE: Clearly state \"So what?\" - business implications. Example: Instead of \"Pearson correlation coefficient = 0.87\", write \"Strong relationship: Higher income leads to higher house prices\". FOR TECHNICAL AUDIENCES: (1) DETAILED: Include confidence intervals, sample sizes, statistical test results. (2) REPRODUCIBLE: Document data sources, transformations, parameters. (3) DIAGNOSTIC: Show residuals, Q-Q plots, diagnostic information. (4) PRECISE: Use technical terms correctly, include model metrics. PUBLICATION-QUALITY REQUIREMENTS: (1) High resolution (300+ DPI for print). (2) Vector graphics (SVG, PDF) when possible - scales without pixelation. (3) Professional color schemes (not default matplotlib). (4) Consistent styling across all figures. (5) Proper citations and data sources. (6) Caption explaining what's shown and key takeaway. (7) Legible when printed in black and white. (8) Fits journal format guidelines. COMMON MISTAKES TO AVOID: (1) Truncated Y-axis misleading magnitude. (2) Too much information in one plot. (3) Poor color choices (rainbow, red-green for colorblind). (4) Missing axis labels or units. (5) Inconsistent scales across subplots. (6) Chart type doesn't match data type. GOLDEN RULE: If you have to explain the plot for 5 minutes, it's too complex. Simplify or break into multiple plots.",
    keyPoints: [
      'Clear purpose: every visualization answers a specific question',
      "Appropriate chart type for data type (don't use pie charts!)",
      'Clean design: maximize data-ink ratio, remove chart junk',
      'Non-technical: simplified, annotated, storytelling approach',
      'Technical: detailed, reproducible, with diagnostic information',
      'Publication-quality: high resolution, vector graphics, proper citations',
    ],
  },
];
