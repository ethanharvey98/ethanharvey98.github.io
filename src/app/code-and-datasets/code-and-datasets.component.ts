import { Component, OnInit } from '@angular/core';
import ProjectsJSON from '../../assets/projects.json';  

interface Project {  
  name: String;
  projects: Boolean;
  code_and_datasets: Boolean;
  page: String;
  image: String;
}

@Component({
  selector: 'app-code-and-datasets',
  templateUrl: './code-and-datasets.component.html',
  styleUrls: ['./code-and-datasets.component.css']
})
export class CodeAndDatasetsComponent implements OnInit {

  projects: Project[] = ProjectsJSON;

  constructor() { }

  ngOnInit(): void {
  }

}
