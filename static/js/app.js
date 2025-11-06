// Initialize map
let map;
let markers = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    checkHealth();
    setupEventListeners();
});

// Initialize Leaflet map centered on Ethiopia
function initializeMap() {
    map = L.map('map').setView([9.1450, 38.7667], 6); // Center on Ethiopia
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);
    
    // Add region markers for major regions
    const regions = [
        { name: 'Addis Ababa', lat: 9.02497, lng: 38.74689, type: 'Capital' },
        { name: 'Tigray', lat: 14.1614, lng: 38.7161, type: 'Region' },
        { name: 'Amhara', lat: 11.7864, lng: 37.7758, type: 'Region' },
        { name: 'Oromia', lat: 8.9806, lng: 38.7578, type: 'Region' },
        { name: 'Somali', lat: 6.6612, lng: 44.0584, type: 'Region' },
        { name: 'Afar', lat: 12.0, lng: 41.5, type: 'Region' },
        { name: 'Dire Dawa', lat: 9.5890, lng: 41.8666, type: 'City' },
        { name: 'Harari', lat: 9.3097, lng: 42.1297, type: 'Region' },
        { name: 'SNNPR', lat: 6.5, lng: 37.5, type: 'Region' },
        { name: 'Gambela', lat: 8.25, lng: 34.5833, type: 'Region' },
        { name: 'Benishangul-Gumuz', lat: 10.5, lng: 35.5, type: 'Region' }
    ];
    
    regions.forEach(region => {
        const marker = L.marker([region.lat, region.lng]).addTo(map);
        marker.bindPopup(`<b>${region.name}</b><br>${region.type}<br><button onclick="askAboutRegion('${region.name}')" style="margin-top: 5px; padding: 5px 10px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">Ask about this region</button>`);
        markers.push(marker);
    });
    
    // Update map info on click
    map.on('click', function(e) {
        updateMapInfo(`Clicked at: ${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`);
    });
}

// Setup event listeners
function setupEventListeners() {
    const input = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendButton');
    
    // Send on Enter key
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize input
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
}

// Check API health
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const statusDot = statusIndicator.querySelector('.status-dot');
        
        if (data.status === 'ready') {
            statusDot.style.background = '#4caf50';
            statusText.textContent = 'Ready';
        } else {
            statusDot.style.background = '#ff9800';
            statusText.textContent = 'Initializing...';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusIndicator = document.getElementById('statusIndicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = document.getElementById('statusText');
        statusDot.style.background = '#f44336';
        statusText.textContent = 'Error';
    }
}

// Send message
async function sendMessage() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    // Add user message to chat
    addMessage(question, 'user');
    
    // Clear input
    input.value = '';
    
    // Show loading
    const loadingId = addLoadingMessage();
    
    // Disable send button
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        
        // Remove loading
        removeLoadingMessage(loadingId);
        
        // Add assistant response
        if (data.error) {
            addMessage(`Error: ${data.error}`, 'assistant', 'error');
        } else {
            addMessage(data.answer, 'assistant', data.intent);
            
            // Show sources if available
            if (data.sources && data.sources.length > 0) {
                showSources(data.sources);
            }
        }
        
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage(`Error: ${error.message}`, 'assistant', 'error');
    } finally {
        sendButton.disabled = false;
        input.focus();
    }
}

// Add message to chat
function addMessage(text, type, intent = null) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Add intent indicator
    let intentBadge = '';
    if (intent === 'price') {
        intentBadge = '<span style="font-size: 0.8em; opacity: 0.8;">💰 Price Data</span><br>';
    } else if (intent === 'context') {
        intentBadge = '<span style="font-size: 0.8em; opacity: 0.8;">🎯 Humanitarian Response</span><br>';
    } else if (intent === 'scenario') {
        intentBadge = '<span style="font-size: 0.8em; opacity: 0.8;">📊 Scenario Analysis</span><br>';
    }
    
    // Format text (preserve line breaks and markdown-like formatting)
    const formattedText = formatText(text);
    
    messageDiv.innerHTML = `<p>${intentBadge}${formattedText}</p>`;
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Format text (basic markdown support)
function formatText(text) {
    // Convert markdown bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert markdown headers
    text = text.replace(/^### (.*$)/gm, '<h4 style="margin: 10px 0 5px 0; font-size: 1.1em;">$1</h4>');
    text = text.replace(/^## (.*$)/gm, '<h3 style="margin: 15px 0 10px 0; font-size: 1.2em;">$1</h3>');
    
    // Convert line breaks
    text = text.replace(/\n/g, '<br>');
    
    // Convert bullet points
    text = text.replace(/^[-*] (.*$)/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>)/s, '<ul style="margin: 10px 0; padding-left: 20px;">$1</ul>');
    
    return text;
}

// Add loading message
function addLoadingMessage() {
    const messagesContainer = document.getElementById('chatMessages');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.id = 'loading-message';
    loadingDiv.innerHTML = `
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        <span>Thinking...</span>
    `;
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return 'loading-message';
}

// Remove loading message
function removeLoadingMessage(id) {
    const loadingDiv = document.getElementById(id);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Show sources
function showSources(sources) {
    const sourcesPanel = document.getElementById('sourcesPanel');
    const sourcesContent = document.getElementById('sourcesContent');
    
    sourcesContent.innerHTML = '';
    
    sources.forEach((source, index) => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        sourceDiv.innerHTML = `
            <strong>${index + 1}. ${source.source}</strong>
            <span class="source-type">(${source.type})</span>
            <div class="source-preview">${source.preview}</div>
        `;
        sourcesContent.appendChild(sourceDiv);
    });
    
    sourcesPanel.style.display = 'block';
}

// Toggle sources panel
function toggleSources() {
    const sourcesPanel = document.getElementById('sourcesPanel');
    sourcesPanel.classList.toggle('expanded');
}

// Ask quick question
function askQuickQuestion(question) {
    document.getElementById('questionInput').value = question;
    sendMessage();
}

// Ask about region
function askAboutRegion(regionName) {
    const question = `What is the food security situation in ${regionName}?`;
    document.getElementById('questionInput').value = question;
    sendMessage();
}

// Update map info
function updateMapInfo(info) {
    document.getElementById('mapInfo').textContent = info;
}

