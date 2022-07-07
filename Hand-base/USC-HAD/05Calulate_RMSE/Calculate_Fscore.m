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
        fieldName=fileNamePredict(fieldStart:(fieldEnd-1));  %%%��predict����ȡ���ؼ��ֶ�
%         predictLabelName=[dataDirPre,fieldName,'.csv'];
        predictLabel = csvread([dataDirPre, fieldName, '.csv']);  %%%��ȡ��һ��predict�ļ�
        
        for y=1:numberFilesReal
            if ~isempty(strfind(fileListReal(y).name,fieldName)) %%%��real��ƥ��ؼ��ֶ�
            
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
            distanceStart=predictLabel(i,2)-realLabel(i,3);         %%%��ʼ��ľ���
            distanceEnd=predictLabel(i,3)-realLabel(i,4);           %%%������ľ���

            predictLength=predictLabel(i,3)-predictLabel(i,2);         %%%Ԥ��Ķ�������
            realLength=realLabel(i,4)-realLabel(i,3);           %%%ʵ�ʵĶ�������

        %     if((distanceStart*distanceStart)/(realLength*realLength)<1)
        %         A=(distanceStart*distanceStart)/(realLength*realLength);
        %         B=(distanceEnd*distanceEnd)/(realLength*realLength);
        %     end
        %     
        %     if((distanceStart*distanceStart)/(realLength*realLength)>=1)
        %         A=(distanceStart*distanceStart)/(predictLength*predictLength);
        %         B=(distanceEnd*distanceEnd)/(predictLength*predictLength);
        %     end
            if abs(distanceStart) <= seg_threshold     %%%���㿪ʼ���TPֵ
                positives(i)= positives(i)+1;
            end
            if abs(distanceEnd) <= seg_threshold     %%%����������TPֵ
                positives(actionNum+i)= positives(actionNum+i)+1;
            end
            if (abs(distanceStart)|abs(distanceEnd)) > seg_threshold 
                %%%��ʼ��
                if predictLabel(i,2)-realLabel(i-1,4)<= seg_threshold | predictLabel(i,2)-realLabel(i-1,3)<= seg_threshold | predictLabel(i,2)-realLabel(i,4)<= seg_threshold | predictLabel(i,2)-realLabel(i+1,3)<= seg_threshold 
                    detected(i) = detected(i)+1;
                end
                %%%������
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