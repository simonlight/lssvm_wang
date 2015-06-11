package io;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.stream.ImageInputStream;

import com.sun.org.apache.xml.internal.utils.IntVector;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import sun.font.ExtendedTextLabel;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;
public class LossCalculatorNegative {
	
	protected static int[] getPatch(Integer totalPatchNum, Integer h){
		Double root = Math.sqrt(totalPatchNum);
		Integer denominator = root.intValue();
		int patch[] = {h/denominator, h%denominator};
		return patch;
	}
	
	protected ArrayList<Rectangle> getETBB(String ETBBPath){
		
		ArrayList<Rectangle> recs = new ArrayList<Rectangle>();
		try {
			FileReader fr = new FileReader(ETBBPath);
			BufferedReader br = new BufferedReader(fr);
			String line = null;
			while((line = br.readLine()) != null){
				String coors[] = line.split(",");
				int xmin = Integer.parseInt(coors[0]);
				int xmax = Integer.parseInt(coors[1]);
				int ymin = Integer.parseInt(coors[2]);
				int ymax = Integer.parseInt(coors[3]);
				recs.add(new Rectangle(xmin,xmax,xmax-xmin,ymax-ymin));
			}
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("getETBB error");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("getETBB error");
		}
		return recs;

	}
	
	protected static Rectangle getPatchBB(int width, int height, int totalPatchNum, int[] patch){
		int up = (int)(1 + Math.floor(patch[0]*0.1*height));
		int down = (int)Math.floor(up + height*(1-(Math.sqrt(totalPatchNum)-1)/10));
		int left = (int)(1 + Math.floor(patch[1]*0.1*width));
		int right = (int)Math.floor(left + width*(1-(Math.sqrt(totalPatchNum)-1)/10));
		return new Rectangle(left,up,right-left,down-up);
	}
	
	protected double getETIntersectionLossOfAllbb(ArrayList<Rectangle> ETBB, Rectangle PBB){
		double totalSurface = PBB.height*PBB.width;
		for(Rectangle oneBB: ETBB){
			totalSurface += oneBB.height*oneBB.width;			
		}
		
		double totalInterSurface = 0;
		for(Rectangle oneBB: ETBB){
			Rectangle interPart = oneBB.intersection(PBB);
			totalInterSurface += 2*(interPart.height*interPart.width);
		}
		return 1 - (totalInterSurface / totalSurface);
		
	}
	
	protected double getRecSurface(Rectangle r){
		return r.height * r.width;
	}
	
	protected double getETIntersectionLossOfGivenClass(ArrayList<Rectangle> ETBB, Rectangle PBB){
		double maxInterRatio = 0;
		for(Rectangle oneBB: ETBB){
			Rectangle interPart = oneBB.intersection(PBB);
			double interSurface = getRecSurface(interPart);
			double unionSurface = getRecSurface(oneBB) + getRecSurface(PBB) - interSurface;
			double interRatio = interSurface / unionSurface;
			if (interRatio > maxInterRatio){
				maxInterRatio = interRatio;
			}
		}
		
		return 1 - maxInterRatio;
	}
	
	protected double getETIntersectionLoss(BagMIL x, Integer h, String ETBBPath) {
		try {
			Iterator<ImageReader> readers = ImageIO  
                .getImageReadersByFormatName("JPG");  
        ImageReader reader = readers.next();  
        File bigFile = new File(x.getName());  
        ImageInputStream iis = ImageIO.createImageInputStream(bigFile);  
        reader.setInput(iis, true);
		int	width = reader.getWidth(0);
		int height = reader.getHeight(0);
		int totalPatchNum = x.getFeatures().size();
		iis.close();
		int[] patch = getPatch(totalPatchNum, h);
		ArrayList<Rectangle> ETBB = getETBB(ETBBPath);
		Rectangle PBB = getPatchBB(width, height, totalPatchNum, patch);
		//calculate total surface as denominator
		return getETIntersectionLossOfGivenClass(ETBB, PBB);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println(x.getName() + "is bad");
			e.printStackTrace();
			return -10000.0;
		}  
	}
	
	
	protected static ArrayList<Point> getGazes(String ETGazesPath){
		
		ArrayList<Point> gazes = new ArrayList<Point>();
		try {
			FileReader fr = new FileReader(ETGazesPath);
			BufferedReader br = new BufferedReader(fr);
			String line = null;
			while((line = br.readLine()) != null){
				String coors[] = line.split(",");
				int x = Integer.parseInt(coors[0]);
				int y = Integer.parseInt(coors[1]);
				gazes.add(new Point(x,y));
			}
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.err.println("getGaze error");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.err.println("getGaze error");
		}
		return gazes;

	}
	
	protected static double ETRatioLossOfGivenClass(String ETGazesPath,Rectangle PBB){
		ArrayList<Point> gazes = getGazes(ETGazesPath);
		double gazeNumber = 0;
		double inGazeNumber = 0;
		for(Point p: gazes){
			if (PBB.contains(p)){
				inGazeNumber+=1;
			}
			gazeNumber+=1;
		}
		System.out.println(1-inGazeNumber/gazeNumber);
		return 1-inGazeNumber/gazeNumber;
	}
	
	protected static double getETRatioLoss(String imagePath, int totalPatchNum, Integer h, String ETGazesPath) {
		try {
			Iterator<ImageReader> readers = ImageIO  
                .getImageReadersByFormatName("JPG");  
        ImageReader reader = readers.next();  
//        File bigFile = new File(x.getName());  
        File bigFile = new File(imagePath);  
        ImageInputStream iis = ImageIO.createImageInputStream(bigFile);  
        reader.setInput(iis, true);  
		int	width = reader.getWidth(0);
		int height = reader.getHeight(0);
//		int totalPatchNum = x.getFeatures().size();
		iis.close();
		int[] patch = getPatch(totalPatchNum, h);
		//ArrayList<Rectangle> ETBB = getETBB(ETGazesPath);
		Rectangle PBB = getPatchBB(width, height, totalPatchNum, patch);
		//calculate total surface as denominator
		return ETRatioLossOfGivenClass(ETGazesPath, PBB);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println(imagePath + "is bad");
			e.printStackTrace();
			return -10000.0;
		}  
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
//		String featurePath[] = x.getFileFeature(h).split("/");	
//		///home/wangxin/Data/ferrari_data//POETdataset/POETdataset/matconvnet_m_2048_features/90/2010_000691_1_1.txt
//		String ETLossFileName = featurePath[featurePath.length - 1];
//		//2010_000691_1_1.txt
//		String ETLossFileName = 
//		String imageFileName[] = x.getName().split("/");
//		///home/wangxin/Data/ferrari_data/POETdataset/POETdataset/PascalImages/sofa_2010_000691.jpg
//		String imClass = imageFileName[imageFileName.length - 1].split("_")[0];
//		//String ETLossPath =  root + "ETLoss_ratio/"+ imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
//		String negative_ETLossPath =  root + "/negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
//		///home/wangxin/Data/ferrari_data/reduit_allbb/negative_ETLoss_ratio/bicycle/sofa/4/2010_000691_1_1.txt
//		
//		//x.getFeatures().size() = 4
//		//String ETBBPath = root + "gt_bb/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 
//		String ETGazesPath = root + "gazes/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
//		String negative_ETGazesPath = root + "gaze_negative/" + objectClass+"/"+imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
//		File negative_f = new File(negative_ETLossPath);
//		negative_f.getAbsoluteFile().getParentFile().mkdirs();
		
		String root = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
		String imageRoot="/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/PascalImages/";
//		String[] classList={"cat", "dog", "bicycle", "motorbike", "boat", "aeroplane", "horse", "cow", "sofa", "diningtable"};
//		String[] classList={"aeroplane", "horse", "cow", "sofa", "diningtable"};
//		String[] classList={"cat", "dog", "bicycle", "motorbike", "boat"};
//		String[] classList={"cat"};
//		String[] classList={"dog"};
//		String[] classList={"bicycle"};
//		String[] classList={"motorbike"};
//		String[] classList={"boat"};
//		String[] classList={"aeroplane"};
		String[] classList={"horse"};
//		String[] classList={"sofa"};
//		String[] classList={"diningtable"};
//		String[] classList={"cow"};
		int[] sizeList = {1,4,9,16,25,36};
		//int[] sizeList = {1};
		File fImageRoot = new File(imageRoot);
		File[] imageList = fImageRoot.listFiles();
		//Class actuel
		for (String objectClass: classList){
			for (int totalPatchNumber: sizeList){
				if (totalPatchNumber !=1){
					for(int h=0;h<totalPatchNumber;h++){
						int x = (int)(h/Math.sqrt(totalPatchNumber));
						int y = (int)(h%Math.sqrt(totalPatchNumber));
						for (File fImage: imageList){
							String imagePath = fImage.getAbsolutePath();
							///home/wangxin/Data/ferrari_data/POETdataset/POETdataset/PascalImages/sofa_2010_000691.jpg
							String imageFileName[] = imagePath.split("/");
							String imageName = imageFileName[imageFileName.length - 1];
							// image class, just a name, if two or more classes are there, negative loss value will be the same since 
							//negative class fixations are aggregated.
							String imClass = imageName.split("_")[0];
							String imCode = imageName.split("_")[1]+"_"+imageName.split("_")[2];
							
							String[] pureNameS = imCode.split("\\.");
							String pureName = pureNameS[0];
							String ETLossFileName = pureName+"_"+String.valueOf(x)+"_"+String.valueOf(y)+".txt";
							String negative_ETLossPath =  root + "/negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+String.valueOf(totalPatchNumber)+"/"+ETLossFileName;
							
							File negative_f = new File(negative_ETLossPath);
							negative_f.getAbsoluteFile().getParentFile().mkdirs();
							
							String negative_ETGazesPath = root + "gaze_negative/" + objectClass+"/"+imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
							
							if(!negative_f.exists()) { 	
								double negative_ETLoss = getETRatioLoss(imagePath, totalPatchNumber, h, negative_ETGazesPath);
								try {					
									BufferedWriter out = new BufferedWriter(new FileWriter(negative_f));
									out.write(String.valueOf(negative_ETLoss));
									out.flush();
									out.close();
								} catch (IOException e) {
								e.printStackTrace();
								System.err.println("No ETLoss file");
								}
							}
						}
					}
				}
				else if (totalPatchNumber ==1){
//					int x = (int)(h/Math.sqrt(totalPatchNumber));
//					int y = (int)(h%Math.sqrt(totalPatchNumber));
					int h=0;
					for (File fImage: imageList){
						String imagePath = fImage.getAbsolutePath();
						///home/wangxin/Data/ferrari_data/POETdataset/POETdataset/PascalImages/sofa_2010_000691.jpg
						String imageFileName[] = imagePath.split("/");
						String imageName = imageFileName[imageFileName.length - 1];
						String imClass = imageName.split("_")[0];
						String imCode = imageName.split("_")[1]+"_"+imageName.split("_")[2];
						
						String[] pureNameS = imCode.split("\\.");
						String pureName = pureNameS[0];
						String ETLossFileName = pureName+".txt";
						String negative_ETLossPath =  root + "/negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+String.valueOf(totalPatchNumber)+"/"+ETLossFileName;
						File negative_f = new File(negative_ETLossPath);
						negative_f.getAbsoluteFile().getParentFile().mkdirs();
						
						String negative_ETGazesPath = root + "gaze_negative/" + objectClass+"/"+imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
						
						if(!negative_f.exists()) { 	
							double negative_ETLoss = getETRatioLoss(imagePath, totalPatchNumber, h, negative_ETGazesPath);
							try {					
								BufferedWriter out = new BufferedWriter(new FileWriter(negative_f));
								out.write(String.valueOf(negative_ETLoss));
								out.flush();
								out.close();
							} catch (IOException e) {
							e.printStackTrace();
							System.err.println("No ETLoss file");
							}
						}
					}
				}
			}
		}
		

	
		
	}
		


		
		
		
		
		
		
		


}
