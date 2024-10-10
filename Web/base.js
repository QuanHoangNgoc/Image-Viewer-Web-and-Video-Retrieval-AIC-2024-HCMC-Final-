const textContainer = document.getElementById("debug");


document.getElementById('userInput').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        // Prevent default form submission if in a form
        event.preventDefault();
        // Call a function or execute code here
        // console.log('User Input Submitted: ', this.value);
        
        // Display the value inside the displayText div
        const userInput = document.getElementById('userInput').value;
        document.getElementById('Query').textContent = userInput || "Please enter some text.";
        // Optionally, clear the input after submission
        this.value = '';
    }
});


document.getElementById('csvFileInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const csvContent = e.target.result;
            upload_CSV(csvContent); 
        };
        reader.readAsText(file);
    }
});


// Back button functionality to reset to show all images
document.getElementById('backButton').addEventListener('click', function () {
    click_Back(); 
});


document.getElementById('showHistogram').addEventListener('click', function () {
    click_Histogram(); 
});


document.getElementById('searchButton').addEventListener('click', function() {
    click_Search();
}); 


document.getElementById('clearButton').addEventListener('click', function() {
    click_Clear();
}); 


document.getElementById('csvButton').addEventListener('click', function() {
    textContainer.textContent = "---Finish---"; 
    const textBox = document.getElementById('mappingTextBox').value; 
    
    let csvText = ""; 
    let cnt = 0; 
    if(textBox.includes(',')) {
        csvText = textBox + "\n"; 
        cnt += 1; 
    }
    imagePaths.forEach(imagePath => {
        if(mappingTextResult(imagePath) !== textBox && cnt < 100) {
            csvText += mappingTextResult(imagePath) + "\n"; 
            cnt += 1;
        }
    });
    
    downloadCSV(csvText, EXPORT_FILE); 
}); 


document.getElementById("rangeNumberSelector").addEventListener("change", function() {
    const selectedNumber = this.value; 
    DIFF = selectedNumber / 2; 
    DIFF = Math.ceil(DIFF); 
});


document.getElementById("imageNumberSelector").addEventListener("change", function() {
    changeImageNumber(); 
});




