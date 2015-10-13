package Test;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Set;

public class ferrari_loss {

	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
		// TODO Auto-generated method stub
		String filename = "/home/wangxin/Data/ferrari_gaze/ETLoss_dict/ETLOSS+_100.loss";
	    ObjectInputStream is = new ObjectInputStream(new FileInputStream(filename));  
        HashMap<String , Double>  temp = (HashMap<String, Double> ) is.readObject();// 从流中读取User的数据  
        is.close();
        Iterator<String> iter = temp.keySet().iterator();
        System.out.println(temp);
        while(iter.hasNext()){
        	String k = iter.next();
        	temp.replace(k,1-temp.get(k));
        }
        ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(filename));  
    	os.writeObject(temp);
    	os.close();
	}

}
