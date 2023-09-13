document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        document.getElementById('send-button').click();
    }
});