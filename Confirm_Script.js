// Container Variables
const container = document.getElementById('confirm-full-wipe-container');
const confirmButton = document.getElementById('confirm-button');
const cancelButton = document.getElementById('cancel-button');
// Delete Button Variables
const DeleteButton = document.getElementById('deletion-button');
// Show Confirmation container when delete button is double clicked
function showConfirmation() {
    if(container.style.display === 'none' || container.style.display === '' || container.style.opacity === '0' || container.style.opacity === '') {
        container.style.display = 'flex';
        container.style.opacity = '1';
    } else if(container.style.display === 'flex' || container.style.display === '' || container.style.opacity === '1' || container.style.opacity === '') {
        container.style.display = 'none';
        container.style.opacity = '0';
    }
    console.log(`Confirmation succesfully shown/hidden.`);
}

// Hide Confirmation when cancel button is clicked
cancelButton.addEventListener('click', function() {
    if(container.style.display === 'flex' || container.style.display === '' || container.style.opacity === '1' || container.style.opacity === '') {
        container.style.display = 'none';
        container.style.opacity = '0';
    }
});