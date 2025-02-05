const clearChatButton = document.getElementById('clear-button');
const saveButtonC = document.getElementById('save-button');
const uploadButton = document.getElementById('load-button');
const logoutButton = document.getElementById('logout-button');

const effectWait1 = '125';
const effectWait2 = '450';

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Function to add effects to a button
function addButtonEffects(button) {
    button.addEventListener('mouseover', () => {
        if(button.style.boxShadow === '0 4.25px 0px rgb(100, 100, 100)' || button.style.boxShadow === '' || button.style.transform === 'translateY(0px)' || button.style.transform === '') {
            button.style.boxShadow = '0 2.25px 0px rgb(100, 100, 100)';
            button.style.transform = 'translateY(2px)';
            wait(effectWait1).then(() => {
                button.style.border = '3px solid rgb(0, 234, 255)';
                wait(effectWait2).then(() => {
                    button.style.borderTop = '3px solid rgb(179, 179, 180)';
                    button.style.borderBottom = '3px solid rgb(179, 179, 180)';
                    button.style.borderLeft = '3px solid rgb(179, 179, 180)';
                    button.style.borderRight = '3px solid rgb(179, 179, 180)';
                });
            });
        } else {
            return;
        }
    });

    button.addEventListener('mouseout', () => {
        if(button.style.boxShadow === '0 2.25px 0px rgb(100, 100 ,100)' || button.style.boxShadow === '' || button.style.transform === 'translateY(2px)' || button.style.transform === '') {
            button.style.boxShadow = '0 4.25px 0px rgb(100, 100, 100)';
            button.style.transform = 'translateY(0px)';
        } else {
            return;
        }
    });

    button.addEventListener('click', () => {
        button.style.border = '3px solid rgb(0, 234, 255)';
        if(button.style.boxShadow === '0 2.25px 0px rgb(100, 100, 100)' || button.style.boxShadow === '' || button.style.transform === 'translateY(3px)' || button.style.transform === '') {
            button.style.boxShadow = '0 1.25px 0px rgb(100, 100, 100)';
        }
    });
}

// Apply effects to both buttons
addButtonEffects(clearChatButton);
addButtonEffects(saveButtonC);
addButtonEffects(uploadButton);
addButtonEffects(logoutButton);
