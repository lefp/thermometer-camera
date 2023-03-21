# thermometer-camera

Using computer vision to automatically read the temperature off a thermometer in the [COSI](https://github.com/COSI-Lab) server room.

Current progress: minimal testing shows it correctly outputs temperature.

Next steps:
1. Mess with the temperature a little to verify that it notices when the temperature deviates.
2. Handle different lighting conditions, so that the light can be switched off. This requires a significant rewrite.
3. Set up a Discord bot that notifies the COSI Discord when the temperature deviates too much.

![image](https://user-images.githubusercontent.com/70862148/226531616-1a3c7fa6-2a44-4501-aec8-b230fa9dad3a.png)
