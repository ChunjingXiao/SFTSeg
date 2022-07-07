clear
    actionNum=7; 
    seg_threshold= 32;         
dataDir = ['E:\AWIFI_Segment2020\ESPRESSO_DataPreprocess\00BlendData\SFT-Seg-upload\hand_base\usc-had\05Calulate_RMSE'];
countFscore(dataDir,actionNum,seg_threshold);

function countFscore(dataDir,actionNum,seg_threshold)
    countNum=0;
    F_score1=0;
    dataDirPre=[dataDir, '\InputLabel_predict\siameseTestCsi_'];
    dataDirReal=[dataDir, '\InputLabel_real\real_'];
    fileListPre = dir(strcat([dataDir, '\InputLabel_predict\'],'*.csv'));
    fileListReal = dir(strcat([dataDir, '\InputLabel_real\'],'*.xls'));
    numberFilesPre = length(fileListPre);
    numberFilesReal = length(fileListReal);
    
    for x=1:numberFilesPre
        fileNamePredict=fileListPre(x).name;
        fieldStart=strfind(fileNamePredict,'usc_');
        fieldEnd=strfind(fileNamePredict,'.csv');
        fieldName=fileNamePredict(fieldStart:(fieldEnd-1));  %%%从predict中提取出关键字段
%         predictLabelName=[dataDirPre,fieldName,'.csv'];
        predictLabel = csvread([dataDirPre, fieldName, '.csv']);  %%%读取第一个predict文件
        
        for y=1:numberFilesReal
            if ~isempty(strfind(fileListReal(y).name,fieldName)) %%%从real中匹配关键字段
            
            realLabel = xlsread([dataDirReal, fieldName,'.xls']);
            countNum=countNum+1;
%         predictLabel = csvread([dataDir, '\InputLabel_predict\'],'siameseTestCsiuser1_rp_6.csv');
%         realLabel = xlsread([dataDir, '\InputLabel_real\'],'real_Deep_1_rp_6.xls']);

            predictLength=eye(1,actionNum);
            realLength=eye(1,actionNum);

            distanceStart=eye(1,actionNum);
            distanceEnd=eye(1,actionNum);


            sumA=0;
            sumB=0;
            totalLength=0;


            P=0; F_score=0; R=0;          %%%F-score=2*P*R/P+R      P=TP/TP+FP     R=TP/TP+FN
            FP=0; FN=0;                 
            positives(1:2*actionNum)=0;
            detected(1:2*actionNum)=0;


        for i=1:actionNum
            distanceStart=predictLabel(i,2)-realLabel(i,3);         %%%开始点的距离
            distanceEnd=predictLabel(i,3)-realLabel(i,4);           %%%结束点的距离

            predictLength=predictLabel(i,3)-predictLabel(i,2);         %%%预测的动作长度
            realLength=realLabel(i,4)-realLabel(i,3);           %%%实际的动作长度

        %     if((distanceStart*distanceStart)/(realLength*realLength)<1)
        %         A=(distanceStart*distanceStart)/(realLength*realLength);
        %         B=(distanceEnd*distanceEnd)/(realLength*realLength);
        %     end
        %     
        %     if((distanceStart*distanceStart)/(realLength*realLength)>=1)
        %         A=(distanceStart*distanceStart)/(predictLength*predictLength);
        %         B=(distanceEnd*distanceEnd)/(predictLength*predictLength);
        %     end
            if abs(distanceStart) <= seg_threshold     %%%计算开始点的TP值
                positives(i)= positives(i)+1;
            end
            if abs(distanceEnd) <= seg_threshold     %%%计算结束点的TP值
                positives(actionNum+i)= positives(actionNum+i)+1;
            end
            if (abs(distanceStart)|abs(distanceEnd)) > seg_threshold 
                %%%开始点
                if predictLabel(i,2)-realLabel(i-1,4)<= seg_threshold | predictLabel(i,2)-realLabel(i-1,3)<= seg_threshold | predictLabel(i,2)-realLabel(i,4)<= seg_threshold | predictLabel(i,2)-realLabel(i+1,3)<= seg_threshold 
                    detected(i) = detected(i)+1;
                end
                %%%结束点
                if predictLabel(i,3)-realLabel(i,3)<= seg_threshold | predictLabel(i,3)-realLabel(i-1,4)<= seg_threshold | predictLabel(i,3)-realLabel(i+1,3)<= seg_threshold | predictLabel(i,3)-realLabel(i+1,4)<= seg_threshold 
                    detected(i) = detected(i)+1;
                end  
            end
        end 

        FN = sum(positives==0);
        FP = sum(detected>0);
        TP = sum(positives>0);

        R = TP / (TP + FN);
        P = TP  / (TP + FP);
        if (P==0 && R==0)
            F_score=0;
        else
            F_score = 2 *((P * R)/(P + R));
        end
        F_score1=F_score1+F_score;
        %disp(F_score);
        end
        end
    end
    F_score1=F_score1/countNum;
    disp(F_score1);
end