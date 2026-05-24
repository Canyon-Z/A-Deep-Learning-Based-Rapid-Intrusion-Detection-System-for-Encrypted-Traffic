/* Wrapper module exposing chart functions and globals from main.js */
(function(){
    window.chartsModule = window.chartsModule || {};

    window.chartsModule.initCharts = function(){ return (window.initCharts ? window.initCharts() : undefined); };
    window.chartsModule.initModelEvalCharts = function(){ return (window.initModelEvalCharts ? window.initModelEvalCharts() : undefined); };
    window.chartsModule.initModalChart = function(){ return (window.initModalChart ? window.initModalChart() : undefined); };
    window.chartsModule.updateTrendChart = function(isMalware){ return (window.updateTrendChart ? window.updateTrendChart(isMalware) : undefined); };

    // expose chart globals
    window.chartsModule.getLineChart = function(){ return window.uploadTrendChart || window.liveTrendChart || null; };
    window.chartsModule.getPieChart = function(){ return window.uploadPieChart || null; };
    window.chartsModule.getTrendLabels = function(){ return window.trendLabels; };
})();
