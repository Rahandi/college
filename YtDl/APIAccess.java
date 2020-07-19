package youtube_downloader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import javax.swing.JOptionPane;
import org.json.JSONException;
import org.json.JSONObject;

public class APIAccess {
	String query = null;
	
	public APIAccess(String que) {
		query = que;
	}
	
	public JSONObject get() throws JSONException {
		JSONObject result = new JSONObject();
		try {
			URL url = new URL("http://rahandiapi.herokuapp.com/youtubeapi?key=randi123&q=" + query);
			URLConnection conn = url.openConnection();
			InputStream input = conn.getInputStream();
			String stri = getStringFromInputStream(input);
			result = new JSONObject(stri);
		}
		catch (Exception e) {
			result.put("error", e.toString());
			e.printStackTrace();
		}
		return result;
	}
	
	private static String getStringFromInputStream(InputStream is) {
		BufferedReader br = null;
		StringBuilder sb = new StringBuilder();
	
		String line;
		try {
			br = new BufferedReader(new InputStreamReader(is));
			while ((line = br.readLine()) != null) {
				sb.append(line);
			}
	
		} catch (IOException e) {
			e.printStackTrace();
			JOptionPane.showMessageDialog(null, "unknown error");
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
					JOptionPane.showMessageDialog(null, "unknown error");
				}
			}
		}
	
		return sb.toString();
	}
}
