/* Wrapper exposing misc helpers from main.js */
(function(){
    window.helpersModule = window.helpersModule || {};
    window.helpersModule.persistLiveUiState = function(){ return (window.persistLiveUiState ? window.persistLiveUiState() : undefined); };
    window.helpersModule.getSelectedInterface = function(){ return (window.getSelectedInterface ? window.getSelectedInterface() : ''); };
    window.helpersModule.cacheRecordedRow = function(kind, entry){ return (window.cacheRecordedRow ? window.cacheRecordedRow(kind, entry) : undefined); };
    window.helpersModule.renderRecordedRows = function(kind){ return (window.renderRecordedRows ? window.renderRecordedRows(kind) : undefined); };
    window.helpersModule.clearLiveResultTable = function(message){ return (window.clearLiveResultTable ? window.clearLiveResultTable(message) : undefined); };
    window.helpersModule.applyRecordFilter = function(filterType){ return (window.applyRecordFilter ? window.applyRecordFilter(filterType) : undefined); };
    window.helpersModule.updateStats = function(data){ return (window.updateStats ? window.updateStats(data) : undefined); };
})();
