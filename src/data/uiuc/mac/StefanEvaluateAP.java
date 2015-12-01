package data.uiuc.mac;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import latent.LatentRepresentation;
import latent.lssvm.multiclass.LSSVMMulticlassFastBagMILET;
import latent.variable.BagMIL;
import struct.STrainingSample;
import data.io.BagReader;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class StefanEvaluateAP {
	
	public static void main(String[] args) {
		
		String dataSource= "big";//local or other things
		String gazeType = "stefan";

		String sourceDir = new String();
		String resDir = new String();

		if (dataSource=="local" && gazeType == "stefan"){
			sourceDir = "/local/wangxin/Data/full_stefan_gaze/";
			resDir = "/local/wangxin/results/full_stefan_gaze/std_et/";
			gazeType = "stefan";
			
		}
		else if (dataSource=="big" && gazeType == "stefan"){
			sourceDir = "/home/wangxin/Data/full_stefan_gaze/";
			resDir = "/home/wangxin/results/full_stefan_gaze/std_et/";
			gazeType = "stefan";
			
		}
	
		String initializedType = ".";//+0,+-,or other things
		boolean hnorm = false;
		
		String taskName = "java_std_et_basic_loss/";
		
		String resultFolder = resDir+taskName;
		
		String resultFilePath = resultFolder + "ap_summary.txt";
		String metricFolder = resultFolder + "metric/";
		String classifierFolder = resultFolder + "classifier/";
		String scoreFolder = resultFolder + "score/";
	
		String[] classes = {args[0]};
		int[] scaleCV = {Integer.valueOf(args[1])};
//		String[] classes = {"jumping" ,"phoning" ,"playinginstrument" ,"reading" ,"ridingbike" ,"ridinghorse" ,"running" ,"takingphoto", "usingcomputer" ,"walking"};
//		String [] classes = {"horse"};
//		String[] classes = {"aeroplane", "cow" ,"dog", "cat" ,"motorbike", "boat" , "horse" , "sofa" ,"diningtable" ,"bicycle"};
//		int[] scaleCV = {90,80,70,60};
//		int[] scaleCV = {50,40};
	    double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {1e-3};
	
//	    double[] tradeoffCV = {0,0.1,0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0};
	    double[] tradeoffCV = {0,0.1,0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0};
	//	    String[] classes = {"dog", "cat", "motorbike", "boat" ,"aeroplane" ,"horse" ,"cow" ,"sofa", "diningtable" ,"bicycle"};
	    
		int optim = 1;
		int epochsLatentMax = 5000;
		int epochsLatentMin = 2;
		int cpmax = 5000;
		int cpmin = 2;
		int numWords = 2048;
	    
		System.out.println("experiment detail: "
				+ "\nsourceDir:\t "+sourceDir
				+ "\nresDir:\t"+resDir
				+ "\ngaze type:\t"+gazeType
				+ "\ninitilaize type:\t"+initializedType
				+ "\nhnorm:\t"+Boolean.toString(hnorm)
				+ "\ntask name:\t"+taskName
				+ "\nclasses CV:\t"+Arrays.toString(classes)
				+ "\nscale CV:\t"+Arrays.toString(scaleCV)
				+ "\nlambda CV:\t" + Arrays.toString(lambdaCV)
				+ "\nepsilon CV:\t" + Arrays.toString(epsilonCV)
				+ "\ntradeoff CV:\t"+Arrays.toString(tradeoffCV)
				+ "\noptim:\t"+optim
				+ "\nepochsLatentMax:\t"+epochsLatentMax
				+ "\nepochsLatentMin:\t"+epochsLatentMin
				+ "\ncpmax:\t"+cpmax
				+ "\ncpmin:\t"+cpmin
				+ "\nnumWords:\t"+numWords
			    );
	    
		
	    for(String className: classes){
	    	for(int scale : scaleCV) {
	    		String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
				List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(listTrainPath, numWords, dataSource); 
	    		String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valval_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
				List<TrainingSample<BagMIL>> listVal = BagReader.readBagMIL(listValPath, numWords, dataSource); 
	    		String listTestPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
				List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(listTestPath, numWords, dataSource); 
					
				for(double epsilon : epsilonCV) {
			    	for(double lambda : lambdaCV) {
			    		for(double tradeoff : tradeoffCV){    		    			
	
			    			List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
			    			for(int i=0; i<listTrain.size(); i++) {
			    				exampleTrain.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTrain.get(i).sample,0), listTrain.get(i).label));    			
			    			}

			    			List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleVal = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();	
			    			for(int i=0; i<listVal.size(); i++) {
			    				exampleVal.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listVal.get(i).sample,0), listVal.get(i).label));
			    			}

			    			List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
							for(int i=0; i<listTest.size(); i++) {
								exampleTest.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTest.get(i).sample,0), listTest.get(i).label));    			
							}
							
							LSSVMMulticlassFastBagMILET lsvm = new LSSVMMulticlassFastBagMILET();
							
							File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
									className + "_" + scale + "_"+epsilon+"_"+lambda + 
									"_"+tradeoff+"_"+cpmax+"_"+cpmin+"_"+epochsLatentMax+"_"+epochsLatentMin +
									"_"+optim+"_"+numWords+".lssvm");
							ObjectInputStream ois;
							System.out.println("read classifier " + fileClassifier.getAbsolutePath());
							try {
								ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
								lsvm = (LSSVMMulticlassFastBagMILET) ois.readObject();
								lsvm.showParameters();
							} catch (FileNotFoundException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							} catch (ClassNotFoundException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
							//train metric file
							File trainMetricFile=new File(metricFolder+scale+"/metric_train_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
							trainMetricFile.getAbsoluteFile().getParentFile().mkdirs();
		    				double ap_train = lsvm.testAPRegion(exampleTrain, trainMetricFile);
		    				System.out.println("ap train:"+ap_train);
							//val metric file
							File valMetricFile=new File(metricFolder+scale+"/metric_valval_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
							valMetricFile.getAbsoluteFile().getParentFile().mkdirs();
		    				double ap_val = lsvm.testAPRegion(exampleVal, valMetricFile);
		    				System.out.println("ap val:"+ap_val);
		    				//test metric file		    				
		    				File testMetricFile=new File(metricFolder+scale+"/metric_valtest_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
		    				testMetricFile.getAbsoluteFile().getParentFile().mkdirs();
		    				double ap_test = lsvm.testAPRegion(exampleTest, testMetricFile);
		    				System.out.println("ap test:"+ap_test);
		    				
		    				//write ap 
		    				try {
								BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
								out.write("category:"+className+" tradeoff:"+String.valueOf(tradeoff)+" scale:"+scale+" lambda:"+lambda+" epsilon:"+epsilon+" ap_train:"+ap_train+" ap_val:"+ap_val+" ap_test:"+ap_test+"\n");
								out.flush();
								out.close();
								
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
		    				System.err.format("train:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_train, className, scale, lambda, epsilon); 
		    				System.err.format("val:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_val, className, scale, lambda, epsilon); 
		    				System.err.format("test:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_test, className, scale, lambda, epsilon); 

						}
			    	}
			    }
			}
		}
	}
	
	
}
