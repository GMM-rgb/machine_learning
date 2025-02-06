document.addEventListener('DOMContentLoaded', () => {
    let settings = document.getElementById('settings-menu-button');
    let settingsMenu = document.getElementById('settings-menu');
    let settingsMenuClose = document.getElementById('settings-menu-close-button');

    settings.addEventListener('click', () => {
        if(settingsMenu.style.display === 'none' || settingsMenu.style.display === '') {
            settingsMenu.style.display = 'flex';
            return;
        } else if(settingsMenu.style.display === 'flex' || settingsMenu.style.display === '') {
            settingsMenu.style.display = 'none';
            return;
        }
    });

    settingsMenuClose.addEventListener('click', () => {
        settingsMenu.style.display = 'none';
    });

    let settingsMenuItems = document.querySelectorAll('.settings-menu-item');
    settingsMenuItems.forEach((item) => {
        item.addEventListener('click', () => {
            if(item.id === 'settings-menu-close-button') {
                settingsMenu.style.display = 'none';
                return;
            } else {
                return;
            }
        });
    });

    const AdvancedGeneration = document.getElementById('advanced-generation-toggle');
    const AdvancedGenerationMenu = document.getElementById('advanced-generation-menu');
    const AdvancedGenerationClose = document.getElementById('advanced-generation-close-button');
    const AdvancedGenerationMenuOpen = document.getElementById('advanced-generation-menu-open-button');
    let advancedGenerationState = false; // Flag to control the advanced generation feature & allow the user to toggle the menu open and closed
    let allowMenuOpen = false; // Flag to control the menu open and close state

    AdvancedGeneration.addEventListener('click', () => {
        if(advancedGenerationState === false && AdvancedGeneration.innerHTML === 'Advanced Generation: Off') {
            AdvancedGeneration.innerHTML = 'Advanced Generation: On';
            advancedGenerationState = true;
            return;
        } else if(advancedGenerationState === true && AdvancedGeneration.innerHTML === 'Advanced Generation: On') {
            AdvancedGeneration.innerHTML = 'Advanced Generation: Off';
            advancedGenerationState = false;
            return;
        } else {
            return;
        }
    });

    AdvancedGenerationClose.addEventListener('click', () => {
        AdvancedGenerationMenu.style.display = 'none';
    });

    if(advancedGenerationState === false) {
        AdvancedGenerationMenu.style.display = 'none';
        allowMenuOpen = false;
        return;
    } else if(advancedGenerationState === true) {
        allowMenuOpen = true;
        return;
    }
    if(allowMenuOpen === false) {
        AdvancedGenerationMenu.style.display = 'none';
        return;
    } else if(allowMenuOpen === true) {
        AdvancedGenerationMenuOpen.addEventListener('click', () => {
            if(AdvancedGenerationMenu.style.display === 'none' || AdvancedGenerationMenu.style.display === '') {
                AdvancedGenerationMenu.style.display = 'flex';
                return;
            } else if(AdvancedGenerationMenu.style.display === 'flex' || AdvancedGenerationMenu.style.display === '') {
                AdvancedGenerationMenu.style.display = 'none';
                return;
            } else {
                return;
            }
        });
        return;
    }
});
