/* Wrapper module exposing live-capture related functions from main.js */
(function(){
    window.liveModule = window.liveModule || {};
    window.liveModule.fetchInterfaces = function(){ return (window.fetchInterfaces ? window.fetchInterfaces() : undefined); };
    window.liveModule.fetchAndUpdateMetrics = function(){ return (window.fetchAndUpdateMetrics ? window.fetchAndUpdateMetrics() : undefined); };
    window.liveModule.pollLive = function(){ return (window.pollLive ? window.pollLive() : undefined); };
    window.liveModule.startLive = function(){ return (window.startLive ? window.startLive() : undefined); };
    window.liveModule.setLiveRunningUI = function(running, options){ return (window.setLiveRunningUI ? window.setLiveRunningUI(running, options) : undefined); };
    window.liveModule.switchToLive = function(){ return (window.switchToLive ? window.switchToLive() : undefined); };
    window.liveModule.switchToUpload = function(){ return (window.switchToUpload ? window.switchToUpload() : undefined); };
})();
