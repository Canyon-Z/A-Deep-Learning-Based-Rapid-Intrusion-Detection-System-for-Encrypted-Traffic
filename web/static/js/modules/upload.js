/* Lightweight wrapper module exposing upload-related functions from main.js
   This file does not change behavior; it exposes existing globals as a grouped API. */
(function(){
    window.uploadModule = window.uploadModule || {};

    window.uploadModule.handleFiles = function(files){ return (window.handleFiles ? window.handleFiles(files) : undefined); };
    window.uploadModule.readResponsePayload = function(response){ return (window.readResponsePayload ? window.readResponsePayload(response) : undefined); };
    window.uploadModule.syncModelSelection = function(modelId, modelName){ return (window.syncModelSelection ? window.syncModelSelection(modelId, modelName) : undefined); };
    window.uploadModule.selectModel = function(modelId, modelName){ return (window.selectModel ? window.selectModel(modelId, modelName) : undefined); };
    window.uploadModule.getModelDisplayName = function(modelId){ return (window.getModelDisplayName ? window.getModelDisplayName(modelId) : modelId); };
    window.uploadModule.uploadFilesSequentially = function(files){ return (window.uploadFilesSequentially ? window.uploadFilesSequentially(files) : undefined); };
    window.uploadModule.uploadFile = function(file){ return (window.uploadFile ? window.uploadFile(file) : undefined); };

    // expose currentActiveModel getter/setter
    Object.defineProperty(window.uploadModule, 'currentActiveModel', {
        get: function(){ return window.currentActiveModel; },
        set: function(v){ window.currentActiveModel = v; }
    });
})();
