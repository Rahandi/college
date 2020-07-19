package youtube_downloader;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import org.json.JSONArray;
import org.json.JSONObject;

public class Interface {
	JFrame frame =  new JFrame();
	JPanel mainPanel = new JPanel(new BorderLayout());
	JTextField txtLink = new JTextField(25);
	JLabel lblJudul = null;
	JLabel lblImage = null;
	JScrollPane scrollPanel = null;
	
	public void GO() {
		JLabel lblLink = new JLabel("Link Youtube:");
		JButton btnGo = new JButton("GO!");
		btnGo.addActionListener(new GoListener());
		JPanel awal = new JPanel();
		awal.setPreferredSize(new Dimension(500, 40));
		awal.add(lblLink);
		awal.add(txtLink);
		awal.add(btnGo);
		mainPanel.add(awal, BorderLayout.NORTH);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(BorderLayout.CENTER, mainPanel);
		frame.setSize(500, 500);
		frame.setTitle("Youtube Downloader");
		frame.getRootPane().setDefaultButton(btnGo);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	
	public void GoClicked(String query) {
		try {
			try {
				Container parent = lblJudul.getParent();
				parent.remove(lblJudul);
				parent.remove(lblImage);
				parent.remove(scrollPanel);
				parent.revalidate();
				parent.repaint();
			}
			catch (Exception e) {}
			APIAccess API = new APIAccess(query);
			ExtractData Data = new ExtractData(API.get());
			String error = Data.cekData();
			if(!"null".equals(error)) {
				JOptionPane.showMessageDialog(frame, error);
				return;
			}
			lblJudul = new JLabel(Data.getTitle());
			lblImage = new JLabel(new ImageIcon(new ImageIcon(Data.getThumbnail()).getImage().getScaledInstance(320, 180, Image.SCALE_SMOOTH)));
			lblJudul.setHorizontalAlignment(SwingConstants.CENTER);
			lblImage.setPreferredSize(new Dimension(320, 180));
			scrollPanel = makeScroll(Data);
			JPanel akhir = new JPanel();
			akhir.setPreferredSize(new Dimension(500, 460));
			akhir.add(lblJudul);
			akhir.add(lblImage);
			akhir.add(scrollPanel);
			mainPanel.add(akhir, BorderLayout.CENTER);
			mainPanel.revalidate();
			mainPanel.repaint();
		}
		catch (Exception e) {
			JOptionPane.showMessageDialog(frame, e.toString());
			e.printStackTrace();
		}
	}
	
	public JScrollPane makeScroll(ExtractData Data) {
		JPanel dummy = new JPanel();
		dummy.setBackground(Color.white);
		dummy.setPreferredSize(new Dimension(300, 250));
		dummy.setAutoscrolls(true);
		JScrollPane dummyScroll = new JScrollPane(dummy);
		dummyScroll.setPreferredSize(new Dimension(400, 200));
		try {
			JSONArray result = Data.getDownload();
			dummy.setLayout(new GridLayout(result.length(), 1));
			JSONObject data = null;
			String url = null;
			String ext = null;
			String reso = null;
			String size = null;
			JLabel labelling = null;
			JButton donlotBtn = null;
			for(int i=0; i<result.length(); i++) {
				data = result.getJSONObject(i);
				url = data.getString("url");
				ext = data.getString("extension");
				reso = data.getString("resolution");
				size = data.getString("size");
				labelling = new JLabel("<html>Extension: " + ext + "<br>Quality: " + reso + "<br>Size: " + size + "</html>");
				donlotBtn = new JButton("Download");
				donlotBtn.setSize(new Dimension(25, 25));
				donlotBtn.putClientProperty("url", url);
				donlotBtn.putClientProperty("ext", ext);
				donlotBtn.putClientProperty("reso", reso);
				donlotBtn.addActionListener(new DownloadListener());
				if(i % 2 == 0) {
					labelling.setOpaque(true);
					labelling.setForeground(Color.gray);
				}
				dummy.add(labelling);
				dummy.add(donlotBtn);
			}
		}
		catch (Exception e) {
			JOptionPane.showMessageDialog(frame, e.toString());
			e.printStackTrace();
		}
		return dummyScroll;
	}
	
	public void downloadData(String userpath, String title, String url, String ext, String reso) {
		title= title.replaceAll("([^A-Za-z0-9 ])", "");
		String namafile = title + "[" + reso + "]." + ext;
		String path = userpath + namafile;
		File file = new File(path);
		file.delete();
		try(InputStream input = new URL(url).openStream()){
			Files.copy(input, Paths.get(path));
			JOptionPane.showMessageDialog(frame, namafile + "\nberhasil di download");
		}
		catch (Exception e) {
			String error = "gagal download";
			if(e.toString().contains("403")) {
				error = error + "\nyou are not allowed to download this";
			}
			JOptionPane.showMessageDialog(frame, error);
			e.printStackTrace();
		}
	}
	
	public class GoListener implements ActionListener{
		public void actionPerformed(ActionEvent event) {
			String query= txtLink.getText();
			GoClicked(query);
		}
	}
	
	public class DownloadListener implements ActionListener{
		public void actionPerformed(ActionEvent event) {
			String title = lblJudul.getText();
			String url = (String)((JButton)event.getSource()).getClientProperty("url");
			String ext = (String)((JButton)event.getSource()).getClientProperty("ext");
			String reso = (String)((JButton)event.getSource()).getClientProperty("reso");
			String path = "";
			JFileChooser chooser = new JFileChooser();
			chooser.setCurrentDirectory(new File("."));
			chooser.setDialogTitle("choose now");
			chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
			chooser.setAcceptAllFileFilterUsed(false);
			if(chooser.showOpenDialog(chooser) == JFileChooser.APPROVE_OPTION) {
				path = chooser.getSelectedFile().toString() + "\\";
			}
			if(path.equals("")) {
				return;
			}
			downloadData(path, title, url, ext, reso);
		}
	}
}
