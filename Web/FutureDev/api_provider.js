// URL of the API endpoint
const apiUrl = 'http://127.0.0.1:5000//send-text';

// Function to send text to the API
async function sendTextToApi(text) {
    const textData = { message: text }; // Prepare the data to send
    
    try {
        const response = await fetch(apiUrl, {
            method: 'POST', // Specify the method
            headers: {
                'Content-Type': 'application/json' // Set the content type to JSON
            },
            body: JSON.stringify(textData) // Convert the text data to JSON string
        });
        
        // Check if the response is OK
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        
        const data = await response.json(); // Parse the JSON response from the API
        // console.log('Response from server:', data); // Log the response
        return data; // Return the response data
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
    }
}

// Example usage: Sending text to the API
const userInput = "A people"; // This can be taken from a user input
sendTextToApi(userInput).then(responseText => {
    console.log('Received text from API:', responseText);
});
