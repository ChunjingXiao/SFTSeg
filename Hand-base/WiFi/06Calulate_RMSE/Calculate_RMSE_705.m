clear
    actionNum=10; %%一个文件中的动作数量
dataDir = ['E:\AWIFI_Segment2020\ESPRESSO_DataPreprocess\00BlendData\SFT-Seg-upload\hand_base\wifi\05Calulate_RMSE'];
countRMSE(dataDir,actionNum);

function countRMSE(dataDir,actionNum)
            countNum=0;
            RMSE0=0;
            
            dataDirPre=[dataDir, '\InputLabel_predict\siameseTestCsiuser'];
            dataDirReal=[dataDir, '\InputLabel_real\real_Deep_'];
            fileListPre = dir(strcat([dataDir, '\InputLabel_predict\'],'*.csv'));
            fileListReal = dir(strcat([dataDir, '\InputLabel_real\'],'*.xls'));
            numberFilesPre = length(fileListPre);
            numberFilesReal = length(fileListReal);
            
            for x=1:numberFilesPre
            fileNamePredict=fileListPre(x).name;
            fieldStart=strfind(fileNamePredict,'1_');
            fieldEnd=strfind(fileNamePredict,'.csv');
            fieldName=fileNamePredict(fieldStart:(fieldEnd-1));  %%%从predict中提取出关键字段
%         predictLabelName=[dataDirPre,fieldName,'.csv'];
            predictLabel = csvread([dataDirPre, fieldName, '.csv']);  %%%读取第一个predict文件
        
            for y=1:numberFilesReal
            if ~isempty(strfind(fileListReal(y).name,fieldName)) %%%从real中匹配关键字段
            
            realLabel = xlsread([dataDirReal, fieldName,'.xls']);
            countNum=countNum+1;
            
            
%         predictLabel = csvread(['E:\AWIFI_Segment2020\ESPRESSO_DataPreprocess\00BlendData\05HandGesture_DeepSeg\05Calulate_RMSE\InputLabel_predict\','siameseTestCsiuser1_rp_6.csv']);
%         realLabel = xlsread(['E:\AWIFI_Segment2020\ESPRESSO_DataPreprocess\00BlendData\05HandGesture_DeepSeg\05Calulate_RMSE\InputLabel_real\','real_Deep_1_rp_6.xls']);

            predictLength=eye(1,actionNum);
            realLength=eye(1,actionNum);

            distanceStart=eye(1,actionNum);
            distanceEnd=eye(1,actionNum);


            sumA=0;
            sumB=0;
            totalLength=0;

        for i=1:actionNum
            distanceStart=predictLabel(i,2)-realLabel(i,3);
            distanceEnd=predictLabel(i,3)-realLabel(i,4);

            predictLength=predictLabel(i,3)-predictLabel(i,2);
            realLength=realLabel(i,4)-realLabel(i,3);

            A=(distanceStart*distanceStart);
            B=(distanceEnd*distanceEnd);

            sumA=sumA+A;
            sumB=sumB+B;
            totalLength=totalLength+realLength;
        end 
            sumA=sumA/actionNum;
            sumB=sumB/actionNum;


            RMSE1=sqrt(sumA)/totalLength;
            RMSE2=sqrt(sumB)/totalLength;

            RMSE=(RMSE1+RMSE2)/2;
            RMSE_original=RMSE*actionNum;
            RMSE1=RMSE1+RMSE_original;
%             disp(RMSE);
            end
            end
            end
            RMSE0=RMSE0/countNum;
            disp(RMSE1);
end


