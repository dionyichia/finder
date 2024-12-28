const socket = io();
const messagesContainer = document.getElementById('messages');
const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');

// Handle file uploads
let documents = [];

const fileInput = document.getElementById('file');
const fileChosen = document.getElementById('file-chosen');

const dropZone = document.getElementById('drop-zone');
const dropContent = dropZone.querySelector('.drop-zone-content');
const loadingAnim = dropZone.querySelector('.loading-animation');


fileInput.addEventListener('change', async function() {
    if (this.files && this.files.length > 0) {
        // Hide content and show loading at the start
        dropContent.style.display = 'none';  // Changed from classList.add('hidden')
        loadingAnim.style.display = 'flex';  // Changed from classList.remove('hidden')

        const formData = new FormData();
        formData.append('file', this.files[0]);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (result.success) {
                documents.push({
                    id: result.name,
                    name: this.files[0].name
                });
                updateDocumentsList();
            }
            
            if (result.message) {
                addSystemMessage(result.message);
            }
        } catch (error) {
            addSystemMessage('Error uploading file');
            console.error('Error:', error);
        } finally {
            // Show content and hide loading at the end
            dropContent.style.display = 'flex';  // Changed from classList.remove('hidden')
            loadingAnim.style.display = 'none';  // Changed from classList.add('hidden')
            this.value = '';
        }
    }
});

// Handle sending messages
messageForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    
    if (message) {
        addUserMessage(message);
        showLoadingMessage();
        socket.emit('send_message', { message: message });
        messageInput.value = '';
    }
});

// Handle receiving messages
socket.on('receive_message', (data) => {
    removeLoadingMessage();
    addReplyMessage(data.response);
});

function addUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${message}</p>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addReplyMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message reply-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${message}</p>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Helper functions to add alert messages
alertMessageContainer = messagesContainer.querySelector('.alert-message-container');
const documentsContainer = document.getElementById('documents-list');
const noDocumentsMsg  = document.getElementById('no-documents');

function addAlertMessage(messageDiv) { 
    alertMessageContainer.appendChild(messageDiv);

    // Set timeout to remove the message after 10 seconds
    setTimeout(() => {
        messageDiv.classList.add('fade-out'); // Add fade-out animation
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove(); // Remove the element after animation ends
            }
        }, 1000); // Match the duration of the fade-out animation (1s)
    }, 3000); // Wait 2 seconds before starting fade-out
}

function addSystemMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'alert-message system-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${message}</p>
        </div>
    `;
    addAlertMessage(messageDiv)
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateDocumentsList() {        
    // Remove all existing document items
    const existingItems = documentsContainer.querySelectorAll('.document-item');
    existingItems.forEach(item => item.remove());
    
    if (documents.length === 0) {
        noDocumentsMsg.style.display = 'block'; 
    } else {
        noDocumentsMsg.style.display = 'none';
        
        documents.forEach(doc => {
            const docElement = document.createElement('div');
            docElement.className = 'document-item';

            // Determine the icon to display based on the file type (for now, just PDFs)
            const fileType = doc.name.split('.').pop().toLowerCase();
            const icon = fileType === 'pdf' 
                ? '<img src="../static/pdf_img.png" alt="PDF" class="document-icon">' 
                : '📄'; // Default emoji for other file types

            docElement.innerHTML = `
                ${icon}
                <span class="document-name">${doc.name}</span>
                <button class="delete-btn" data-name="${doc.name}">✕</button>
            `;
            
            const deleteBtn = docElement.querySelector('.delete-btn');
            deleteBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                const docName = deleteBtn.dataset.name;
                try {
                    const response = await fetch(`/delete/${docName}`, {
                        method: 'DELETE'
                    });
                    if (response.ok) {
                        documents = documents.filter(d => d.name !== docName);
                        updateDocumentsList();
                    }
                } catch (error) {
                    console.error('Error deleting document:', error);
                }
            });
            
            documentsContainer.appendChild(docElement);
        });
    }
}

function showLoadingAnimation() {
    noDocumentsMsg.style.display = 'none';
    document.getElementById('loading-animation').classList.remove('hidden');
    document.getElementById('no-documents').classList.add('hidden');
}

function hideLoadingAnimation() {
    document.getElementById('loading-animation').classList.add('hidden');
    if (documents.length === 0) {
        document.getElementById('no-documents').classList.remove('hidden');
    }
}

function dropHandler(ev) {
    ev.preventDefault();

    const dropZone = document.getElementById('drop-zone');
    dropZone.classList.remove('drag-over');

    if (ev.dataTransfer.items) {
        const file = ev.dataTransfer.items[0].getAsFile();

        if (file) {
            // Use DataTransfer to set the file to the input element
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            // Trigger the upload function
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    }
}

function dragOverHandler(ev) {
    ev.preventDefault();
    document.getElementById('drop-zone').classList.add('drag-over');
}

function dragLeaveHandler(ev) {
    document.getElementById('drop-zone').classList.remove('drag-over');
}

function showLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-message';
    loadingDiv.innerHTML = `
        <div class="loading-content">
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();
}

function removeLoadingMessage() {
    const loadingMessage = document.querySelector('.loading-message');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}


// Initialize the documents list
updateDocumentsList();