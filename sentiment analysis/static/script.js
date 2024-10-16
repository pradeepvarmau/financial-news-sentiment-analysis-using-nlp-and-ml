document.getElementById('sentiment-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const text = document.getElementById('text').value;

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<strong>Sentiment Scores:</strong><br>
                               Positive: ${data.pos}<br>
                               Neutral: ${data.neu}<br>
                               Negative: ${data.neg}<br>
                               Compound: ${data.compound}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
