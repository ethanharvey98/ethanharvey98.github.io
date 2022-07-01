import { Component, OnInit } from '@angular/core';
import ProjectsJSON from '../../assets/projects.json';  

interface Project {  
  name: String;
  dataset: Boolean;
  page: String;
  image: String;
}

@Component({
  selector: 'app-projects',
  templateUrl: './projects.component.html',
  styleUrls: ['./projects.component.css']
})
export class ProjectsComponent implements OnInit {

  projects: Project[] = ProjectsJSON;

  constructor() { }

  ngOnInit(): void {
  }

}
