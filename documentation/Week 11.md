
*Started with making spectrogram for the train dataset<br/>
*used the spectrogram to run our CNN model<br/>
*With input as 20 mfcc coefficients, 12 chromas and 128 mSpec we modeled each layer.<br/>
*For mfcc we used 32 filter with 5*5 kernel then a 64 with 3*3 kernel and then 2*2 kernel with 128 filters<br/>
*We used max pooling, batch normalization, ReLu etc for the layer.<br/>
*For the chroma features we used 32 filter with 5*5 kernel and a stride of (1,3) then 64 with 3*3 kernel with stride of (1,2) and then 2*2<br/>
*For the mspec feature 32 filter with 5*5 kernel and stride of (2,3) then 64 with 3*3 kernel with stride (2,2) and then 2*2 kernel<br/>
*Completing the report and started preparing for presentation<br/>

