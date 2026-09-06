import React, { useState, useRef, useEffect } from "react";
import {
  Sparkles,
  Send,
  Layers,
  ZoomIn,
  ZoomOut,
  Sun,
  Moon,
  UploadCloud,
  Home,
  ChevronDown,
  Terminal,
  Bot,
  BarChart3,
  Loader2,
  Waves,
  Building2,
  AlertTriangle,
  Flame,
  FileText,
  HelpCircle,
  Clock,
  Radio,
  Satellite,
  SearchCode,
  Compass,
  XCircle,
  CheckCircle2
} from "lucide-react";

export default function SatQueryApp() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState("Single Image");
  const [query, setQuery] = useState("What natural hazards or environmental conditions are occurring here?");
  const [loading, setLoading] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showTraceModal, setShowTraceModal] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const [error, setError] = useState(null);

  const [image1, setImage1] = useState(null);
  const [file1Name, setFile1Name] = useState("");
  const [image2, setImage2] = useState(null);
  const [file2Name, setFile2Name] = useState("");
  const [clientPreview, setClientPreview] = useState(null);

  // CRITICAL FIX: analysisResult is NULL by default (no zombie data)
  const [analysisResult, setAnalysisResult] = useState(null);

  const fileInputRef1 = useRef(null);
  const fileInputRef2 = useRef(null);

  // Clean up preview URLs on unmount
  useEffect(() => {
    return () => {
      if (clientPreview && clientPreview.startsWith("blob:")) {
        URL.revokeObjectURL(clientPreview);
      }
    };
  }, [clientPreview]);

  const sampleQueries = [
    "What natural hazards or environmental conditions are occurring here?",
    "Describe the land-cover and major objects visible in this image.",
    "Identify water bodies and assess their extent and clarity.",
    "Are there signs of wildfire, smoke, or thermal anomalies?",
    "Detect urban infrastructure and settlement patterns."
  ];

  const handleFileUpload = (e, isSecond = false) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Clear any previous errors
    setError(null);

    // Validate file type
    const validTypes = ['image/tiff', 'image/tif', 'image/png', 'image/jpeg', 'image/jpg'];
    const fileName = file.name.toLowerCase();
    const isValid = validTypes.some(type => file.type === type) ||
                   fileName.endsWith('.tif') ||
                   fileName.endsWith('.tiff') ||
                   fileName.endsWith('.png') ||
                   fileName.endsWith('.jpg') ||
                   fileName.endsWith('.jpeg');

    if (!isValid) {
      setError("Please upload a valid image file (TIFF, PNG, or JPEG)");
      return;
    }

    if (!isSecond) {
      setImage1(file);
      setFile1Name(file.name);

      // Show preview for non-TIFF files
      if (!fileName.endsWith('.tif') && !fileName.endsWith('.tiff')) {
        // Revoke old URL to prevent memory leaks
        if (clientPreview && clientPreview.startsWith("blob:")) {
          URL.revokeObjectURL(clientPreview);
        }
        const url = URL.createObjectURL(file);
        setClientPreview(url);
      } else {
        setClientPreview(null);
      }
    } else {
      setImage2(file);
      setFile2Name(file.name);
    }
  };

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setError(null);
    setZoomLevel(1);
    setHoveredIdx(null);
  };

  const executeAnalysis = async () => {
    // Validation
    if (activeTab === "AutoFetch") {
      setError("AutoFetch Mode: This feature requires API keys for Copernicus/Landsat. Please upload an image directly.");
      return;
    }

    if (!image1) {
      setError("Please upload a satellite image first.");
      return;
    }

    // Clear previous results and errors
    resetAnalysis();
    setLoading(true);

    const formData = new FormData();
    formData.append("mode", activeTab);
    formData.append("query", query);
    formData.append("image1", image1);
    if (image2) formData.append("image2", image2);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();

      // Validate response
      if (!data.title || !data.dynamic_cards) {
        throw new Error("Invalid response format from server");
      }

      setAnalysisResult({
        title: data.title,
        dynamicCards: data.dynamic_cards || [],
        technicalReport: data.technical_report,
        confidenceScore: data.confidence_score,
        previewUrl: data.preview_url,
        features: data.features || [],
        classDistribution: data.class_distribution || [],
        spectralMetrics: data.spectral_metrics || {},
        executionTrace: data.execution_summary,
        domain: data.domain
      });

      // Clear the staging preview since we now have results
      if (clientPreview && clientPreview.startsWith("blob:")) {
        URL.revokeObjectURL(clientPreview);
        setClientPreview(null);
      }

    } catch (err) {
      console.error("Analysis error:", err);
      setError(err.message || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getCardIcon = (type, category = "") => {
    const text = category.toLowerCase();
    if (text.includes("fire") || text.includes("burn") || text.includes("flame") || text.includes("smoke")) {
      return <Flame className="w-4 h-4 text-rose-500" />;
    }
    if (type === "water" || text.includes("water") || text.includes("hydro") || text.includes("river") || text.includes("lake")) {
      return <Waves className="w-4 h-4 text-sky-500" />;
    }
    if (type === "urban" || text.includes("urban") || text.includes("infrastructure") || text.includes("building") || text.includes("land cover")) {
      return <Building2 className="w-4 h-4 text-indigo-400" />;
    }
    return <AlertTriangle className="w-4 h-4 text-amber-500" />;
  };

  const getDomainBadge = (domain) => {
    const badges = {
      "WILDFIRE_HAZARD": { icon: <Flame className="w-3 h-3" />, label: "Wildfire Hazard", color: "bg-rose-500/10 text-rose-400 border-rose-500/30" },
      "COASTAL_MARINE": { icon: <Waves className="w-3 h-3" />, label: "Coastal/Marine", color: "bg-blue-500/10 text-blue-400 border-blue-500/30" },
      "URBAN_LANDSCAPE": { icon: <Building2 className="w-3 h-3" />, label: "Urban Landscape", color: "bg-purple-500/10 text-purple-400 border-purple-500/30" },
      "TERRESTRIAL_LANDSCAPE": { icon: <Compass className="w-3 h-3" />, label: "Terrestrial", color: "bg-green-500/10 text-green-400 border-green-500/30" }
    };

    const badge = badges[domain] || badges["TERRESTRIAL_LANDSCAPE"];
    return (
      <span className={`text-xs px-2.5 py-1 rounded-full font-medium border flex items-center gap-1.5 ${badge.color}`}>
        {badge.icon}
        {badge.label}
      </span>
    );
  };

  const isPairedMode = activeTab === "Bi-Temporal Change" || activeTab === "Optical + SAR Pair";

  return (
    <div className={`min-h-screen flex ${darkMode ? "bg-[#0B0F19] text-gray-100" : "bg-[#F8FAFC] text-slate-900"}`}>
      {/* Sidebar */}
      <aside className={`w-64 border-r flex flex-col justify-between p-5 shrink-0 ${
        darkMode ? "bg-[#111827] border-gray-800" : "bg-white border-slate-200 shadow-sm"
      }`}>
        <div>
          <div className="flex items-center gap-3 px-1 py-2 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-blue-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className={`font-bold text-base tracking-tight leading-tight ${darkMode ? "text-white" : "text-slate-900"}`}>
                SatQuery AI
              </h1>
              <span className={`text-[11px] font-semibold tracking-wide ${darkMode ? "text-gray-400" : "text-indigo-600"}`}>
                v16.0 • Auditable
              </span>
            </div>
          </div>

          <nav className="space-y-1.5">
            <button className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-bold transition-colors ${
              darkMode ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/30" : "bg-indigo-50 text-indigo-700 border border-indigo-100"
            }`}>
              <Home className="w-4 h-4" /> Analysis Console
            </button>
            <button
              onClick={() => setShowTraceModal(true)}
              disabled={!analysisResult}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
                analysisResult
                  ? darkMode
                    ? "text-gray-300 hover:bg-gray-800/60 hover:text-white"
                    : "text-slate-600 hover:bg-slate-100"
                  : "text-gray-600 cursor-not-allowed opacity-50"
              }`}
            >
              <Terminal className="w-4 h-4 text-emerald-500" /> Execution Trace
            </button>

            {analysisResult && (
              <button
                onClick={resetAnalysis}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
                  darkMode ? "text-amber-400 hover:bg-amber-500/10 border border-amber-500/20" : "text-amber-700 hover:bg-amber-50 border border-amber-200"
                }`}
              >
                <XCircle className="w-4 h-4" /> Clear Results
              </button>
            )}
          </nav>

          {error && (
            <div className={`mt-4 p-3 rounded-lg border text-xs ${
              darkMode ? "bg-rose-500/10 border-rose-500/30 text-rose-400" : "bg-rose-50 border-rose-200 text-rose-700"
            }`}>
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`w-full flex items-center justify-between p-3 rounded-xl border text-xs font-bold transition-all ${
            darkMode
              ? "bg-[#161F30] border-gray-700 text-gray-200 hover:bg-gray-800"
              : "bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200"
          }`}
        >
          <div className="flex items-center gap-2">
            {darkMode ? <Moon className="w-4 h-4 text-indigo-400" /> : <Sun className="w-4 h-4 text-amber-500" />}
            <span>{darkMode ? "Dark Console" : "Light Mode"}</span>
          </div>
          <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
        </button>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        <div className="p-8 max-w-7xl w-full mx-auto space-y-6">
          {/* Controls Panel */}
          <div className={`p-6 rounded-2xl border transition-all ${
            darkMode ? "bg-[#111827] border-gray-800 shadow-xl" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <h2 className={`text-lg font-bold flex items-center gap-2 ${darkMode ? "text-white" : "text-slate-900"}`}>
                Geospatial Intelligence Platform
                <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${darkMode ? "bg-indigo-950 text-indigo-300 border border-indigo-800" : "bg-indigo-50 text-indigo-700"}`}>
                  Context-Aware Grounding
                </span>
              </h2>

              {/* Mode Selector */}
              <div className="flex gap-1.5 p-1 rounded-xl bg-black/10 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                {[
                  { id: "Single Image", label: "Single Image", icon: null },
                  { id: "Bi-Temporal Change", label: "Change Detection", icon: <Clock className="w-3 h-3" /> },
                  { id: "Optical + SAR Pair", label: "Optical+SAR", icon: <Radio className="w-3 h-3" /> },
                  { id: "AutoFetch", label: "AutoFetch", icon: <Satellite className="w-3 h-3" /> }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setActiveTab(tab.id);
                      resetAnalysis();
                      setError(null);
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                      activeTab === tab.id
                        ? darkMode
                          ? "bg-indigo-600 text-white shadow"
                          : "bg-white text-indigo-700 shadow-sm"
                        : darkMode
                        ? "text-gray-400 hover:text-gray-200"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Bar */}
            <div className={`flex items-center rounded-xl border p-1.5 mb-3 transition-colors ${
              darkMode ? "bg-[#0B0F19] border-gray-700 focus-within:border-indigo-500" : "bg-slate-50 border-slate-300 focus-within:border-indigo-600"
            }`}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !loading && executeAnalysis()}
                placeholder="Ask questions about fires, water bodies, urban areas, vegetation, or environmental hazards..."
                className={`w-full bg-transparent px-3 py-2 text-xs font-medium outline-none ${
                  darkMode ? "text-white placeholder-gray-500" : "text-slate-900 placeholder-slate-400"
                }`}
                disabled={loading}
              />
              <button
                onClick={executeAnalysis}
                disabled={loading || !image1}
                className={`px-5 py-2.5 rounded-lg shadow font-bold text-xs flex items-center gap-2 shrink-0 transition-all ${
                  loading || !image1
                    ? "bg-gray-600 cursor-not-allowed opacity-50"
                    : "bg-indigo-600 hover:bg-indigo-500 text-white"
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Analyze
                  </>
                )}
              </button>
            </div>

            {/* Suggested Queries */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <span className={`text-[11px] font-semibold flex items-center gap-1 ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                <HelpCircle className="w-3.5 h-3.5" /> Sample Queries:
              </span>
              {sampleQueries.map((sq, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(sq)}
                  disabled={loading}
                  className={`text-[11px] px-2.5 py-1 rounded-md border transition-all truncate max-w-xs ${
                    darkMode
                      ? "bg-gray-800/80 border-gray-700 text-gray-300 hover:border-indigo-500 hover:text-white disabled:opacity-50"
                      : "bg-slate-100 border-slate-200 text-slate-700 hover:border-indigo-400 hover:text-indigo-800 disabled:opacity-50"
                  }`}
                  title={sq}
                >
                  {sq}
                </button>
              ))}
            </div>

            {/* Ingestion Area */}
            <div className={`pt-4 border-t ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
              {activeTab === "AutoFetch" ? (
                <div className={`p-4 rounded-xl border ${darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-slate-50 border-slate-200"}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-amber-400 flex items-center gap-2">
                      <SearchCode className="w-4 h-4" /> AutoFetch Mode (Configuration Required)
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 font-mono font-bold">
                      API Keys Needed
                    </span>
                  </div>
                  <p className={`text-xs mb-3 ${darkMode ? "text-gray-400" : "text-slate-600"}`}>
                    AutoFetch requires API credentials for Copernicus Open Access Hub and USGS EarthExplorer. For this demo, please upload images directly using Single Image mode.
                  </p>
                </div>
              ) : (
                <div className={`grid gap-4 ${isPairedMode ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
                  <div
                    onClick={() => !loading && fileInputRef1.current.click()}
                    className={`p-3.5 rounded-xl border-2 border-dashed flex items-center gap-3 transition-all ${
                      loading ? "cursor-not-allowed opacity-50" : "cursor-pointer"
                    } ${
                      file1Name
                        ? darkMode ? "border-indigo-500/50 bg-indigo-950/10" : "border-indigo-400 bg-indigo-50/30"
                        : darkMode ? "border-gray-700 hover:border-indigo-500 bg-[#0B0F19]" : "border-slate-300 hover:border-indigo-600 bg-slate-50"
                    }`}
                  >
                    <input
                      ref={fileInputRef1}
                      type="file"
                      accept=".tif,.tiff,.png,.jpg,.jpeg"
                      onChange={(e) => handleFileUpload(e, false)}
                      className="hidden"
                      disabled={loading}
                    />
                    <UploadCloud className="w-5 h-5 text-indigo-500 shrink-0" />
                    <div className="truncate flex-1">
                      <p className={`text-xs font-bold truncate ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                        {file1Name || "Upload Satellite Image"}
                      </p>
                      <p className={`text-[11px] ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                        {file1Name ? "✓ Ready for analysis" : "TIFF, GeoTIFF, PNG, or JPEG"}
                      </p>
                    </div>
                    {file1Name && (
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                    )}
                  </div>

                  {isPairedMode && (
                    <div
                      onClick={() => !loading && fileInputRef2.current.click()}
                      className={`p-3.5 rounded-xl border-2 border-dashed flex items-center gap-3 transition-all ${
                        loading ? "cursor-not-allowed opacity-50" : "cursor-pointer"
                      } ${
                        file2Name
                          ? darkMode ? "border-indigo-500/50 bg-indigo-950/10" : "border-indigo-400 bg-indigo-50/30"
                          : darkMode ? "border-gray-700 hover:border-indigo-500 bg-[#0B0F19]" : "border-slate-300 hover:border-indigo-600 bg-slate-50"
                      }`}
                    >
                      <input
                        ref={fileInputRef2}
                        type="file"
                        accept=".tif,.tiff,.png,.jpg,.jpeg"
                        onChange={(e) => handleFileUpload(e, true)}
                        className="hidden"
                        disabled={loading}
                      />
                      {activeTab === "Bi-Temporal Change" ? (
                        <Clock className="w-5 h-5 text-amber-500 shrink-0" />
                      ) : (
                        <Radio className="w-5 h-5 text-emerald-500 shrink-0" />
                      )}
                      <div className="truncate flex-1">
                        <p className={`text-xs font-bold truncate ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                          {file2Name || (activeTab === "Bi-Temporal Change" ? "Upload Date 2 Image" : "Upload SAR Image")}
                        </p>
                        <p className={`text-[11px] ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                          {file2Name ? "✓ Secondary loaded" : "Required for paired analysis"}
                        </p>
                      </div>
                      {file2Name && (
                        <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* RESULTS DISPLAY - Only shows after analysis */}
          {analysisResult ? (
            <div className={`p-6 rounded-2xl border transition-all ${
              darkMode ? "bg-[#111827] border-gray-800 shadow-xl" : "bg-white border-slate-200 shadow-sm"
            }`}>
              <div className={`flex flex-wrap justify-between items-center gap-3 mb-4 pb-3 border-b ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-indigo-400 flex items-center gap-1.5 uppercase tracking-wide">
                    <Bot className="w-4 h-4" /> Analysis Complete
                  </span>
                  {analysisResult.domain && getDomainBadge(analysisResult.domain)}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-emerald-400 font-bold px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                    Confidence: {analysisResult.confidenceScore}
                  </span>
                </div>
              </div>

              <h3 className={`text-xl font-bold mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
                {analysisResult.title}
              </h3>

              {/* Dynamic Assessment Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                {analysisResult.dynamicCards.map((card, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-xl border ${
                      darkMode ? "bg-[#0E1726] border-gray-800" : "bg-slate-50 border-slate-200 shadow-sm"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {getCardIcon(card.type, card.category)}
                      <span className={`text-xs font-bold ${darkMode ? "text-gray-200" : "text-slate-800"}`}>
                        {card.category}
                      </span>
                    </div>
                    <p className={`text-xs leading-relaxed ${darkMode ? "text-gray-300" : "text-slate-600"}`}>
                      {card.text}
                    </p>
                  </div>
                ))}
              </div>

              {/* Technical Intelligence Report */}
              <div className={`p-4 rounded-xl border mb-6 ${
                darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-slate-50 border-slate-200"
              }`}>
                <div className="flex items-center gap-2 mb-2 text-xs font-bold uppercase tracking-wider text-indigo-400">
                  <FileText className="w-3.5 h-3.5" /> Synthesized Intelligence Report
                </div>
                <p className={`text-xs leading-relaxed whitespace-pre-line ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                  {analysisResult.technicalReport}
                </p>
              </div>

              {/* Spectral Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                {Object.entries(analysisResult.spectralMetrics).map(([k, v], idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-xl border ${
                      darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-white border-slate-200 shadow-sm"
                    }`}
                  >
                    <span className={`text-[10px] block uppercase font-mono font-bold mb-1 ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                      {k}
                    </span>
                    <span className={`text-sm font-bold ${v === "N/A" ? "text-gray-500" : "text-indigo-400"}`}>
                      {v}
                    </span>
                  </div>
                ))}
              </div>

              {/* Viewport + Vector Grounding */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                <div className={`lg:col-span-8 rounded-2xl overflow-hidden border relative bg-black ${
                  darkMode ? "border-gray-800" : "border-slate-200"
                }`}>
                  <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-black/75 backdrop-blur-md p-1.5 rounded-xl border border-white/10 shadow-lg">
                    <button
                      onClick={() => setZoomLevel((z) => Math.min(z + 0.2, 2.5))}
                      className="p-1.5 text-gray-300 hover:text-white transition-colors"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setZoomLevel((z) => Math.max(z - 0.2, 0.5))}
                      className="p-1.5 text-gray-300 hover:text-white transition-colors"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setShowOverlays(!showOverlays)}
                      className={`p-1.5 rounded-lg transition-colors ${
                        showOverlays ? "text-indigo-400 bg-indigo-500/20" : "text-gray-400"
                      }`}
                      title="Toggle Polygons"
                    >
                      <Layers className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div
                    className="relative w-full h-[450px] overflow-hidden"
                    style={{
                      transform: `scale(${zoomLevel})`,
                      transformOrigin: "center center",
                      transition: "transform 0.2s ease-out"
                    }}
                  >
                    <img
                      src={analysisResult.previewUrl}
                      alt="Satellite Scene"
                      className="w-full h-full object-cover select-none"
                    />

                    {showOverlays && analysisResult.features?.length > 0 && (
                      <svg
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox="0 0 1024 1024"
                        preserveAspectRatio="none"
                      >
                        {analysisResult.features.map((f, idx) => {
                          const isHovered = hoveredIdx === idx;
                          return (
                            <g key={f.id || idx} className="transition-all duration-300">
                              <polygon
                                points={f.points}
                                fill={isHovered ? `${f.color}88` : `${f.color}40`}
                                stroke={f.color}
                                strokeWidth={isHovered ? "5" : "3"}
                                className="transition-all duration-200"
                              />
                              {f.center && (
                                <>
                                  <circle
                                    cx={f.center[0]}
                                    cy={f.center[1]}
                                    r={isHovered ? "20" : "16"}
                                    fill={f.color}
                                    stroke="#FFFFFF"
                                    strokeWidth="3"
                                    className="transition-all duration-200"
                                  />
                                  <text
                                    x={f.center[0]}
                                    y={f.center[1] + 5}
                                    textAnchor="middle"
                                    fill="#FFFFFF"
                                    fontSize={isHovered ? "14" : "13"}
                                    fontWeight="bold"
                                    className="transition-all duration-200"
                                  >
                                    {idx + 1}
                                  </text>
                                </>
                              )}
                            </g>
                          );
                        })}
                      </svg>
                    )}
                  </div>
                </div>

                {/* Grounded Distribution */}
                <div className="lg:col-span-4 space-y-3">
                  <div className={`flex items-center justify-between pb-2 border-b ${darkMode ? "border-gray-800" : "border-slate-200"}`}>
                    <h4 className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                      <BarChart3 className="w-3.5 h-3.5 text-indigo-400" /> Class Distribution
                    </h4>
                    <span className={`text-[10px] font-mono font-bold ${darkMode ? "text-gray-500" : "text-slate-400"}`}>
                      {analysisResult.classDistribution?.length || 0} Classes
                    </span>
                  </div>

                  {analysisResult.classDistribution?.map((item, idx) => {
                    const isHovered = hoveredIdx === idx;
                    return (
                      <div
                        key={idx}
                        onMouseEnter={() => setHoveredIdx(idx)}
                        onMouseLeave={() => setHoveredIdx(null)}
                        className={`p-3 rounded-xl border transition-all cursor-pointer ${
                          isHovered
                            ? darkMode
                              ? "border-indigo-500 bg-indigo-500/10 shadow-lg scale-[1.02]"
                              : "border-indigo-500 bg-indigo-50 shadow-md scale-[1.02]"
                            : darkMode
                            ? "border-gray-800 bg-[#0B0F19] hover:border-gray-700"
                            : "border-slate-200 bg-white hover:border-slate-300 shadow-sm"
                        }`}
                      >
                        <div className="flex justify-between items-center text-xs mb-1.5">
                          <span className={`flex items-center gap-2 font-bold ${darkMode ? "text-white" : "text-slate-800"}`}>
                            <span
                              className="w-4 h-4 rounded-md text-[10px] font-bold text-white flex items-center justify-center shrink-0"
                              style={{ backgroundColor: item.color }}
                            >
                              {idx + 1}
                            </span>
                            <span className="truncate max-w-[160px]" title={item.name}>
                              {item.name}
                            </span>
                          </span>
                          <span className={`font-mono text-xs font-bold ${darkMode ? "text-gray-300" : "text-slate-700"}`}>
                            {item.percentage}%
                          </span>
                        </div>

                        <div className={`w-full rounded-full h-2 overflow-hidden mb-2 ${darkMode ? "bg-gray-800" : "bg-slate-100"}`}>
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${item.percentage}%`, backgroundColor: item.color }}
                          />
                        </div>

                        {item.desc && (
                          <p className={`text-[11px] leading-snug ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                            {item.desc}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            /* Standby State - Clean slate before analysis */
            <div className={`p-12 rounded-2xl border text-center flex flex-col items-center justify-center min-h-[400px] transition-all ${
              darkMode ? "bg-[#111827] border-gray-800" : "bg-white border-slate-200 shadow-sm"
            }`}>
              {clientPreview ? (
                <div className="max-w-md w-full space-y-4">
                  <div className="rounded-xl overflow-hidden border border-gray-700 relative h-64 bg-black shadow-xl">
                    <img src={clientPreview} alt="Staged Preview" className="w-full h-full object-cover" />
                    <div className="absolute top-2 right-2 px-2 py-1 rounded-lg bg-black/75 backdrop-blur-sm text-[10px] font-mono text-emerald-400 border border-emerald-500/30">
                      STAGED
                    </div>
                  </div>
                  <p className="text-xs font-bold text-emerald-400 flex items-center justify-center gap-2">
                    <CheckCircle2 className="w-4 h-4" />
                    {file1Name}
                  </p>
                  <p className={`text-xs ${darkMode ? "text-gray-400" : "text-slate-600"}`}>
                    Image ready. Enter your query above and click <strong className="text-indigo-400">Analyze</strong> to generate context-aware polygons and spectral indices.
                  </p>
                </div>
              ) : (
                <div className="max-w-sm space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto text-indigo-400">
                    <Compass className="w-6 h-6" />
                  </div>
                  <h3 className={`text-sm font-bold ${darkMode ? "text-white" : "text-slate-800"}`}>
                    Geospatial Analysis Engine Ready
                  </h3>
                  <p className={`text-xs leading-relaxed ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                    Upload a satellite image (GeoTIFF, PNG, or JPEG), formulate your analytical question, and execute the pipeline to receive grounded polygon boundaries, spectral indices, and domain-expert synthesis.
                  </p>
                  <div className={`mt-4 p-3 rounded-lg border text-left ${darkMode ? "bg-indigo-500/5 border-indigo-500/20" : "bg-indigo-50 border-indigo-200"}`}>
                    <p className={`text-[11px] font-semibold mb-1 ${darkMode ? "text-indigo-300" : "text-indigo-700"}`}>
                      Supported Queries:
                    </p>
                    <ul className={`text-[11px] space-y-0.5 ${darkMode ? "text-gray-400" : "text-slate-600"}`}>
                      <li>• Wildfire detection & burn severity</li>
                      <li>• Water body extent & quality</li>
                      <li>• Urban expansion & infrastructure</li>
                      <li>• Land cover classification</li>
                      <li>• Environmental hazard assessment</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Execution Trace Modal */}
      {showTraceModal && analysisResult?.executionTrace && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50"
          onClick={() => setShowTraceModal(false)}
        >
          <div
            className={`border p-6 rounded-2xl max-w-2xl w-full shadow-2xl max-h-[80vh] overflow-y-auto ${
              darkMode ? "bg-[#111827] border-gray-700 text-white" : "bg-white border-slate-200 text-slate-900"
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-sm text-indigo-400 flex items-center gap-2">
                <Terminal className="w-4 h-4" /> Auditable Execution Trace
              </h3>
              <div className="flex items-center gap-2">
                <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono border border-emerald-500/20">
                  Transparent AI
                </span>
                <button
                  onClick={() => setShowTraceModal(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Model Pipeline Status */}
            {analysisResult.executionTrace.models && (
              <div className="mb-4 space-y-2">
                <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Model Pipeline</h4>
                {analysisResult.executionTrace.models.map((model, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border flex items-center justify-between ${
                      darkMode ? "bg-[#0B0F19] border-gray-800" : "bg-slate-50 border-slate-200"
                    }`}
                  >
                    <div>
                      <div className="text-xs font-bold text-indigo-300">{model.name}</div>
                      <div className={`text-[11px] ${darkMode ? "text-gray-400" : "text-slate-600"}`}>{model.role}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      {model.status === "Success" || model.status === "Completed" ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                      )}
                      <span className={`text-[11px] font-mono ${
                        model.status === "Success" || model.status === "Completed"
                          ? "text-emerald-400"
                          : "text-amber-400"
                      }`}>
                        {model.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Full JSON Trace */}
            <pre className={`p-4 rounded-xl text-[10px] font-mono overflow-auto max-h-96 border ${
              darkMode ? "bg-[#0B0F19] text-emerald-400 border-gray-800" : "bg-slate-50 text-slate-700 border-slate-200"
            }`}>
              {JSON.stringify(analysisResult.executionTrace, null, 2)}
            </pre>

            <button
              onClick={() => setShowTraceModal(false)}
              className="mt-4 w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold transition-colors"
            >
              Close Trace
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
