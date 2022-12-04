**Semantic Segmentation of Satellite Imagery Project for CS 301 - 103,**
**Group 10: Kamil Arif and Moiez Qamar**

Model:
Info about the model - The distilled model is much smaller than the original full U-Net. The original U-Net has ~23 Convolutional layers and 1.9M trainable parameters, while the distilled model has 8 Convolutional layers and 25.8K trainable parameters. In real terms, this results in the distilled model having a file size of 186 KB, compared to the old model which is 23,002 KB. This is is a 99.2% size reduction with comparable results after a mere 20 epochs of training. 

Training Process:
The training process info - place holder rn

  Precision vs Recall Curve:
![Precision vs Recall](https://cdn.discordapp.com/attachments/610972035195207730/1049032276572373102/image.png)


  Loss vs. Epochs (0-40 epochs):
![Loss vs. Epochs (0-40 epochs)](https://cdn.discordapp.com/attachments/610972035195207730/1049032234134413332/image.png)
  
  ------------------------------------------------------------
  ![image 1](https://cdn.discordapp.com/attachments/610972035195207730/1049043067820326983/image.png)
  ![image 2](https://cdn.discordapp.com/attachments/610972035195207730/1049043131749908490/image.png)
  ![image 3](https://cdn.discordapp.com/attachments/610972035195207730/1049043202830778498/image.png)
  ![image 4](https://cdn.discordapp.com/attachments/610972035195207730/1049043294266593280/image.png)
  ![image 5](https://cdn.discordapp.com/attachments/610972035195207730/1049043570683809922/image.png)
  ![image 6](https://cdn.discordapp.com/attachments/610972035195207730/1049044116023025664/image.png)
  ![image 7](https://cdn.discordapp.com/attachments/610972035195207730/1049044021189812365/image.png)
  ![image 8](https://cdn.discordapp.com/attachments/610972035195207730/1049044363465994260/image.png)
  ![image 9](https://cdn.discordapp.com/attachments/610972035195207730/1049044172180574318/image.png)
  ![image 10](https://cdn.discordapp.com/attachments/610972035195207730/1049043509090459780/image.png)
