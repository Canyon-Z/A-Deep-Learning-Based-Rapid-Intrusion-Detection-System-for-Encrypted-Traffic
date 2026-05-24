/* Wrapper exposing result / modal helpers from main.js */
(function(){
    window.resultsModule = window.resultsModule || {};
    window.resultsModule.addResultRow = function(name, size, time, data, options){ return (window.addResultRow ? window.addResultRow(name,size,time,data,options) : undefined); };
    window.resultsModule.openDetails = function(rowId){ return (window.openDetails ? window.openDetails(rowId) : undefined); };
    window.resultsModule.normalizeDetailPayload = function(data){ return (window.normalizeDetailPayload ? window.normalizeDetailPayload(data) : data); };
    window.resultsModule.formatConfidenceText = function(v){ return (window.formatConfidenceText ? window.formatConfidenceText(v) : v); };
    window.resultsModule.formatExecTime = function(v){ return (window.formatExecTime ? window.formatExecTime(v) : v); };
    window.resultsModule.formatCaptureTime = function(v){ return (window.formatCaptureTime ? window.formatCaptureTime(v) : v); };
    window.resultsModule.initModalChart = function(){ return (window.initModalChart ? window.initModalChart() : undefined); };
    window.resultsModule.showDetails = function(){ return (window.showDetails ? window.showDetails.apply(null, arguments) : undefined); };
})();
